import os
import argparse
import json
import random

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

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='/mnt/data-rundong/robot_datasets/tokenizer-training/pizza_width_split/')
parser.add_argument('--dst', type=str, default='/mnt/data-rundong/robot_datasets/tokenizer-training/pizza_preprocessed_for_pie/')

args = parser.parse_args()

for split in ["train", "test"]:
    src_path = args.src
    for i in range(split):
        j = split[i]
        src_filepath = os.path.join(src_path, j, f'{j}.jsonl')
        f = open(src_filepath, 'r')
        dst_filepath = os.path.join(args.dst, j,  f'{j}.jsonl')
        os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
        dst_file = open(dst_filepath, 'w')

        # read and store all lines
        for line in f:
            instance_data = json.loads(line)

            image_root = '/mnt/robotdata/datasets/pizza_robot'
            # prompt_input = '<|image_1|>\n<|image_2|>\n<bott_i>' + instance_data['task_description'] + '<eott_i>'
            task_description = instance_data['task_description']
            # prompt_output_format = '<boa_o>{}<eoa_o>'
            prompt_output_format = '{}'
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

                    stacked_instance = {}
                    stacked_instance["task_description"] = task_description
                    stacked_instance["answer"] = prompt_output_action
                    stacked_instance["image_paths"] = images
                    dst_file.write(json.dumps(stacked_instance) + '\n')
                except:
                    id = instance_data['ID']
                    traj = instance_data['trajectory_id']
                    print(f'Error occurs when processing task: {id}, trajectory: {traj}, start: {start}')
                    continue








                    
    #     lines = f.readlines()
    #     n_lines = len(lines)
    #     line_cnt = -1
    #     while True:
    #         line_cnt += 1
    #         if line_cnt == n_lines:
    #             break
    #         line = lines[line_cnt]
    #         instance_data = json.loads(line)
    #         trajectory_id = instance_data['trajectory_id']
    #         view = instance_data['view']
    #         start_frame = instance_data['start_frame']
    #         end_frame = instance_data['end_frame']

    #         output_vision_tokens = []
    #         output_action_tokens = []
    #         new_line_cnt = line_cnt
    #         cur_start_frame = start_frame
    #         cur_end_frame = end_frame
    #         new_line = lines[new_line_cnt]
    #         new_instance_data = json.loads(new_line)
    #         # output 3 clips
    #         for _ in range(3):  
    #             find_next_clip = False
    #             if cur_start_frame != cur_end_frame: # prev clip is not duplicated tail frames, look for next clip
    #                 for i in range(1, 9):
    #                     # should find the next clip in 3 frames, use 9 for debug
    #                     new_line_cnt += i
    #                     new_line = lines[new_line_cnt]
    #                     new_instance_data = json.loads(new_line)
    #                     new_trajectory_id = new_instance_data['trajectory_id']
    #                     new_view = new_instance_data['view']
    #                     if not (new_trajectory_id == trajectory_id and new_view == view):
    #                         continue
    #                     cur_start_frame = new_instance_data['start_frame']
    #                     if cur_start_frame == cur_end_frame: # find next clip
    #                         output_vision_tokens += new_instance_data['video_tokens']
    #                         output_action_tokens += new_instance_data['action_tokens']
    #                         cur_end_frame = new_instance_data['end_frame']
    #                         find_next_clip = True
    #                         break
    #                 assert(find_next_clip)
    #             else: # prev clip is duplicated tail frames, use it again
    #                 output_vision_tokens += new_instance_data['video_tokens']
    #                 output_action_tokens += new_instance_data['action_tokens']
                    



    #         '''
    #         create a new data that stack these two instances, with the following fields
    #         - trajectory_id: a integer that identifies the trajectory
    #         - view: a string that describes the view
    #         - start_frame: the start frame of the clip, -1 means it is 6 duplicate first frames
    #         - task_description: a string that describes the task, identical for clips with the same trajectory_id
    #         - scene_description: a string that describes the initial scene, identical for clips with the same trajectory_id and view
    #         - input_clip_description: a string that describes the frame difference in the input clip
    #         - output_clip_description: a string that describes the frame difference in the output clip
    #         - input_video_tokens: a 2D array of size 768 (256 * 3),
    #             256 * 3 is because each clip has 6 frames and downsamples by factor 2
    #         - output_video_tokens: a 2D array of size 768 (256 * 3),
    #         - input_action_tokens: a 2D array of size 42 (6 * 7),
    #         - output_action_tokens: a 2D array of size 42 (6 * 7),
    #         '''
    #         stacked_instance = {}
    #         stacked_instance['trajectory_id'] = trajectory_id
    #         stacked_instance['view'] = view
    #         stacked_instance['input_start_frame'] = instance_data['start_frame']
    #         stacked_instance['input_end_frame'] = instance_data['end_frame']
    #         stacked_instance['output_end_frame'] = cur_end_frame
    #         stacked_instance['task_description'] = instance_data['task_description']
    #         stacked_instance['input_video_tokens'] = instance_data['video_tokens']
    #         stacked_instance['output_video_tokens'] = output_vision_tokens
    #         stacked_instance['input_action_tokens'] = instance_data['action_tokens']
    #         stacked_instance['output_action_tokens'] = output_action_tokens
    #         # randomly split 10% data for validation
    #         if random.random() < 0.1:
    #             dst_file_test.write(json.dumps(stacked_instance) + '\n')
    #         else:
    #             dst_file_train.write(json.dumps(stacked_instance) + '\n')

    # print('Stack Finished.')