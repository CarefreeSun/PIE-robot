import json
import os
from datasets import Dataset, DatasetDict, IterableDataset, Dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import glob

def VLA_dataset_generator(shards, eos_token, static_video_description, return_info, action_before_vision, wo_text, wo_vision, only_text):
    '''
    each shard is a jsonl file, with each line containing a json object
    the json object contains the following fields:
    - trajectory_id: a integer that identifies the trajectory
    - view: a string that describes the view
    - start_frame: the start frame of the clip, -1 means it is 6 duplicate first frames
    - task_description: a string that describes the task, identical for clips with the same trajectory_id
    - scene_description: a string that describes the initial scene, identical for clips with the same trajectory_id and view
    - input_clip_description: a string that describes the frame difference in the input clip
    - output_clip_description: a string that describes the frame difference in the output clip
    - input_video_tokens: a 2D array of size 768 (256 * 3),
        256 * 3 is because each clip has 6 frames and downsamples by factor 2
    - output_video_tokens: a 2D array of size 768 (256 * 3),
    - input_action_tokens: a 2D array of size 42 (6 * 7),
    - output_action_tokens: a 2D array of size 42 (6 * 7),
    
    output:
    a generator that yields a dictionary with only the 'text' field

    text = '<bott_i>' + data['task_description'] + '<eott_i>' + \
            '<bots_i>' + data['scene_description'] + '<eots_i>' + \
            '<botp_i>' + data['input_clip_description'] + '<eotp_i>' + \ 
            '<bov_i>' + ''.join([f'<va{str(x)}>' for x in data['input_video_tokens']]) + '<eov_i>' + \
            '<boa_i>' + ''.join([f'<va{str(x)}>' for x in data['input_action_tokens']]) + '<eoa_i>' + \
            '<botp_o>' + data['output_clip_description'] + '<eotp_o>' + \
            '<bov_o>' + ''.join([f'<va{str(x)}>' for x in data['output_video_tokens']]) + '<eov_o>' + \
            '<boa_o>' + ''.join([f'<va{str(x)}>' for x in data['output_action_tokens']) + '<eoa_o>' + eos_token
    length: 14 special tokens + 
            768 * 2 video tokens +
            42 * 2 action tokens +
            200 task description, scene description, input clip, output clip
            2 eos_token and bos_token (will be automatically added by the tokenizer)
            thus, 2048 sequence length is enough
    '''
    def reset_gripper_width(x):
        return '0' if x > 0.07 else '1'
    
    
    def format_action(values, grip):
        formatted_str = '['
    
        for value in values:
            # 四舍五入并乘以10000
            rounded_value = round(value, 4)
            int_value = int(rounded_value * 10000)
            
            # 格式化
            if int_value >= 0:
                formatted_value = f"+{int_value:03d}"
            else:
                formatted_value = f"{int_value:04d}"
            formatted_str += formatted_value + ','

        formatted_str += grip
        formatted_str += ']'
        
        return formatted_str
    

    for shard in shards:
        with open(shard, "r") as f:
            for line in f:

                instance_data = json.loads(line)

                image_root = '/mnt/robotdata/datasets/pizza_robot'
                prompt_input = '<|image_1|>\n<|image_2|>\n<bott_i>' + instance_data['task_description'] + '<eott_i>'
                prompt_output_format = '<boa_o>{}<eoa_o>'
                image_format = image_root + '/' + str(instance_data['ID']) + '/' + str(instance_data['trajectory_id']) + '/images/right_rgb' + '/{:03d}' + '.jpg'

                num_frames = instance_data["frame_number"]
                num_input_interval = 3
                num_pred_actions = 6

                # 去掉最后补全用的重复帧
                prev_frame_id = -100
                for frame_pos in range(num_frames):
                    cur_frame_id = instance_data['image_indices'][frame_pos]
                    if cur_frame_id == prev_frame_id: # 重复
                        num_frames = frame_pos
                        break
                    # 未重复
                    prev_frame_id = cur_frame_id

                num_start = num_frames ###########
                for start in range(-1, num_start):
                    images = []
                    prompt_output_action = ''
                    try:
                        if start == -1:
                            img_start = image_format.format(instance_data['image_indices'][0])
                            if not os.path.exists(img_start):
                                continue
                            images = [img_start] * 2
                            
                            pred_action_start_idx = 0 # 预测的action开始的index，注意是image_indices中的顺序而不是实际的frame_id
                            pred_action_end_idx = pred_action_start_idx + num_pred_actions - 1
                            if pred_action_end_idx >= num_start:
                                continue # 不到一个clip的数据，太短没有意义

                            pred_action_text = ''

                            for pred_action_idx in range(pred_action_start_idx, pred_action_end_idx + 1):
                                pred_xyzrpy_vec = instance_data['actions'][pred_action_idx][:-1]
                                pred_gripper = reset_gripper_width(instance_data['action_gripper'][pred_action_idx][-1])
                                pred_action_text += format_action(pred_xyzrpy_vec, pred_gripper) # e.g. [+000,-005,-001,+050,+002,+007,+788,0]
                                pred_action_text += ','
                            
                            prompt_output_action = prompt_output_format.format(pred_action_text)

                        else:
                            img_start_idx = start
                            img_end_idx = img_start_idx + num_input_interval
                            if img_end_idx >= num_start:
                                continue
                            img_start = image_format.format(instance_data['image_indices'][img_start_idx])
                            img_end = image_format.format(instance_data['image_indices'][img_end_idx])
                            if not os.path.exists(img_start):
                                continue
                            if not os.path.exists(img_end):
                                continue
                            images = [img_start, img_end]

                            pred_action_start_idx = img_end_idx 
                            pred_action_end_idx = pred_action_start_idx + num_pred_actions - 1

                            pred_action_text = ''

                            for pred_action_idx in range(pred_action_start_idx, pred_action_end_idx + 1):
                                if pred_action_idx >= num_start: # 超出边界
                                    pred_xyzrpy_vec = [0. for _ in range(6)]
                                    pred_gripper = 0 # 默认静止，夹爪闭合
                                else:
                                    pred_xyzrpy_vec = instance_data['actions'][pred_action_idx][:-1]
                                    pred_gripper = reset_gripper_width(instance_data['action_gripper'][pred_action_idx][-1])

                                pred_action_text += format_action(pred_xyzrpy_vec, pred_gripper) # e.g. [+000,-005,-001,+050,+002,+007,0]
                                pred_action_text += ','
                                
                            prompt_output_action = prompt_output_format.format(pred_action_text)

                        # prompt_output_action += eos_token
                        yield {"prompt": prompt_input, 
                               "answer": prompt_output_action, 
                               "images": images}
                    except:
                        continue
                


                    # text_input = '<bov_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_video_tokens']]) + '<eov_i>' + '<boa_i><va0><eoa_i>'
                    # text_input += 'What should the robot arm do to ' + instance_data['task_description'] 
                    # text_input += "? Answer: <boa_o>"

                    # text_output = ''.join([f'<va{str(x)}>' for x in instance_data['output_action_tokens']]) + '<eoa_o>'

                        
                    # if only_text: # For debugging: check if can train the language model correctly
                    #     if instance_data['input_clip_description'] == '': # sample a description for the input clip
                    #         instance_data['input_clip_description'] = random.choice(static_video_description)
                    #     text_input = '<bott_i>' + instance_data['task_description'] + '<eott_i>' + \
                    #             '<bots_i>' + instance_data['scene_description'] + '<eots_i>' + \
                    #             '<botp_i>' + instance_data['input_clip_description'] + '<eotp_i>'
                    #     text_output = '<botp_o>' + instance_data['output_clip_description'] + '<eotp_o>'
                    # else: 
                    #     assert wo_text
                    #     if wo_text:
                    #         text_input = '<bott_i>' + instance_data['task_description'] + '<eott_i>'
                    #         text_output = ''
                    #     else:
                    #         if instance_data['input_clip_description'] == '': # sample a description for the input clip
                    #             instance_data['input_clip_description'] = random.choice(static_video_description)
                    #         text_input = '<bott_i>' + instance_data['task_description'] + '<eott_i>' + \
                    #                 '<bots_i>' + instance_data['scene_description'] + '<eots_i>' + \
                    #                 '<botp_i>' + instance_data['input_clip_description'] + '<eotp_i>'
                    #         text_output = '<botp_o>' + instance_data['output_clip_description'] + '<eotp_o>'
                    #         # if len(text_input) > 900 or len(text_output) > 800:
                    #         #     continue
                    #         if len(text_input) > 900:
                    #             text_input = '<bott_i>' + instance_data['task_description'] + '<eott_i>' + \
                    #                 '<bots_i>' + instance_data['scene_description'] + '<eots_i>' + \
                    #                 '<botp_i>' + '' + '<eotp_i>'
                    #         if len(text_output) > 800:
                    #             text_output = '<botp_o>' + '' + '<eotp_o>'
                        
                    #     # text_input += '<bov_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_video_tokens']]) + '<eov_i>' + \
                    #     #             '<boa_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_action_tokens']]) + '<eoa_i>'
                    #     text_input += '<bov_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_video_tokens']]) + '<eov_i>' + '<boa_i><va0><eoa_i>'
                    #     text_output += '<boa_o>' + ''.join([f'<va{str(x)}>' for x in instance_data['output_action_tokens']]) + '<eoa_o>'
                    # text_output += eos_token

                # if return_info:
                #     yield {"input": text_input, "output": text_output, 
                #            "trajectory_id": instance_data['trajectory_id'], "view": instance_data['view'],
                #            "gt_actions": instance_data['gt_actions']}
                # else:
                #     yield {"input": text_input, "output": text_output}
                # yield {"text": text_input + text_output}

def get_VLA_dataset(args, eos_token, split='train', return_info=False):
    if args.data_root is not None:
        root = args.data_root
        shards = glob.glob(os.path.join(root, split, '*.jsonl'))
    elif args.data_roots is not None:
        shards = []
        for root in args.data_roots:
            shards.extend(glob.glob(os.path.join(root, split, '*.jsonl')))
    else:
        assert False, 'data_root or data_roots must be provided'

    # len_shard = len(shards)
    # shards = shards[:len_shard // 2]
 
    if args.data_debug:
        shards = shards[:1]
    if args.dataset_type == 'dataset':
        ds = Dataset.from_generator(VLA_dataset_generator, gen_kwargs={"shards": shards, 
                                                            "eos_token": eos_token,
                                                            "static_video_description": args.static_video_description,
                                                            "return_info": return_info,
                                                            "action_before_vision": args.action_before_vision,
                                                            "wo_text": args.wo_text,
                                                            "wo_vision": args.wo_vision,
                                                            "only_text": args.only_text
                                                            })
    else: # iterable dataset
        ds = IterableDataset.from_generator(VLA_dataset_generator, gen_kwargs={"shards": shards, 
                                                                "eos_token": eos_token,
                                                                "static_video_description": args.static_video_description,
                                                                "return_info": return_info,
                                                                "action_before_vision": args.action_before_vision,
                                                                "wo_text": args.wo_text,
                                                                "wo_vision": args.wo_vision,
                                                                "only_text": args.only_text
                                                                })
        # ds.column_names = ['text']
    return ds