import os
import argparse
import json
import random

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='/mnt/data-rundong/robot_datasets/0715-dinov2-action111-bridge-noMask-woResidual_step25000_tokenized/rt1')
parser.add_argument('--dst', type=str, default='/mnt/data-rundong/robot_datasets/0715-dinov2-action111-bridge-noMask-woResidual_step25000_tokenized_stacked/rt1')

args = parser.parse_args()

for split in ["train", "test"]:
    src_path = args.src
    for i in split:
        src_filepath = os.path.join(src_path, f'{i}.jsonl')
        f = open(src_filepath, 'r')
        dst_filepath = os.path.join(args.dst, f'{i}.jsonl')
        os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
        dst_file = open(dst_filepath, 'w')

        # read and store all lines
                    
        lines = f.readlines()
        n_lines = len(lines)
        line_cnt = -1
        while True:
            line_cnt += 1
            if line_cnt == n_lines:
                break
            line = lines[line_cnt]
            instance_data = json.loads(line)
            trajectory_id = instance_data['trajectory_id']
            view = instance_data['view']
            start_frame = instance_data['start_frame']
            end_frame = instance_data['end_frame']

            output_vision_tokens = []
            output_action_tokens = []
            new_line_cnt = line_cnt
            cur_start_frame = start_frame
            cur_end_frame = end_frame
            new_line = lines[new_line_cnt]
            new_instance_data = json.loads(new_line)
            # output 3 clips
            for _ in range(3):  
                find_next_clip = False
                if cur_start_frame != cur_end_frame: # prev clip is not duplicated tail frames, look for next clip
                    for i in range(1, 9):
                        # should find the next clip in 3 frames, use 9 for debug
                        new_line_cnt += i
                        new_line = lines[new_line_cnt]
                        new_instance_data = json.loads(new_line)
                        new_trajectory_id = new_instance_data['trajectory_id']
                        new_view = new_instance_data['view']
                        if not (new_trajectory_id == trajectory_id and new_view == view):
                            continue
                        cur_start_frame = new_instance_data['start_frame']
                        if cur_start_frame == cur_end_frame: # find next clip
                            output_vision_tokens += new_instance_data['video_tokens']
                            output_action_tokens += new_instance_data['action_tokens']
                            cur_end_frame = new_instance_data['end_frame']
                            find_next_clip = True
                            break
                    assert(find_next_clip)
                else: # prev clip is duplicated tail frames, use it again
                    output_vision_tokens += new_instance_data['video_tokens']
                    output_action_tokens += new_instance_data['action_tokens']
                    



            '''
            create a new data that stack these two instances, with the following fields
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
            '''
            stacked_instance = {}
            stacked_instance['trajectory_id'] = trajectory_id
            stacked_instance['view'] = view
            stacked_instance['input_start_frame'] = instance_data['start_frame']
            stacked_instance['input_end_frame'] = instance_data['end_frame']
            stacked_instance['output_end_frame'] = cur_end_frame
            stacked_instance['task_description'] = instance_data['task_description']
            stacked_instance['input_video_tokens'] = instance_data['video_tokens']
            stacked_instance['output_video_tokens'] = output_vision_tokens
            stacked_instance['input_action_tokens'] = instance_data['action_tokens']
            stacked_instance['output_action_tokens'] = output_action_tokens
            # randomly split 10% data for validation
            if random.random() < 0.1:
                dst_file_test.write(json.dumps(stacked_instance) + '\n')
            else:
                dst_file_train.write(json.dumps(stacked_instance) + '\n')

    print('Stack Finished.')