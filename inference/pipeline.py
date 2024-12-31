import json
import argparse
import torch
import os
import re
from PIL import Image

import os
import argparse
from configs import H4ArgumentParser, DataArguments, ModelArguments
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

import os
import json


def decode_action(action_str: str):
    """
    transform action strings into list of action values
    format of action_str (after remove all space key):
    <boa_o>{action contents}<eoa_o><|end|>\n<|endoftext|>
    action contents: strings like "[+018,-019,+039,-047,+035,+005,0],", 6 clips
    """
    # remove all the space key
    action_str = action_str.replace(" ", "")
    # extract action contents between <boa_o> and <eoa_o>
    match = re.search(r"<boa_o>(.*?)<eoa_o>", action_str, re.DOTALL)

    if match:
        try:
            # remove <boa_o> and <eoa_o>, and the last ","
            list_str = match.group(1).strip()[:-1]
            sublists = list_str.split("],[")
            result = []
            for sublist in sublists:
                # remove "[" and "]", and split values by ","
                elements = sublist.strip("[]").split(",")

                # transform action values
                result.append([int(e) / 10000 for e in elements[:-1]].append(int(elements[-1])))
            return result
        except:
            raise ValueError(f"Wrong action format in prediction: {action_str}")
    else:
        raise ValueError(f"<boa_o> or <eoa_o> not found in prediction: {action_str}")


@torch.no_grad()
def preprocess_data(instance_data: dict, processor: AutoProcessor):
    """
    preprocess instance data for VLM
    instance data contains "task_description" and "image_paths" fields
    """
    images = []
    for i in range(len(instance_data['image_paths'])):
        images.append(Image.open(instance_data['image_paths'][i]))

    prompt_input = '<|image_1|>\n<|image_2|>\n<bott_i>{}<eott_i>'.format(instance_data['task_description'])
    prompt_message = {
        'role': 'user',
        'content': f'{prompt_input}',
    }
    prompt = processor.tokenizer.apply_chat_template(
        [prompt_message], tokenize=False, add_generation_prompt=True
    )

    processed_data = processor(prompt, images, return_tensors="pt")

    return processed_data
    

@torch.no_grad()
def call_vla(instance_data: dict, processor: AutoProcessor, tokenizer: AutoTokenizer, vla_pipe: AutoModelForCausalLM, device):

    input_data = preprocess_data(instance_data, processor).to(device)

    generation_args = { 
        "max_new_tokens": 2048, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = vla_pipe.generate(**input_data, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    # remove input tokens 
    generate_ids = generate_ids[:, input_data['input_ids'].shape[1]:]
    print(generate_ids)
    output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    print(output_text)
    
    output_action_pred = decode_action(output_text)
    output_clip_description_pred = ""
    return output_action_pred, output_clip_description_pred


def call_models(instance_data: dict, processor: AutoProcessor, tokenizer: AutoTokenizer, vla_pipe: AutoModelForCausalLM, device):

    output_action_pred, output_clip_description_pred = call_vla(instance_data, processor, tokenizer, vla_pipe, device)

    instance_data['clip_description'] = output_clip_description_pred
    instance_data['actions'] = output_action_pred

    return instance_data

def call_robot(instance_data, robot) -> dict:
    '''
    use the predicted actions to call the robot
    should override the image_paths with the observation after movement, 
    and override the given actions with the actual ones (as the robot may not be able to follow the predicted actions exactly)
    return the new instance_data 
    '''
    pass

def main():

    parser = H4ArgumentParser((ModelArguments, DataArguments))
    vla_args, data_args = parser.parse()

    local_rank = os.getenv('LOCAL_RANK', 0)
    device = f'cuda:{local_rank}'

    # 0. define the tokenizer, processor and vla model
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(vla_args.model_name_or_path, trust_remote_code=True)
    # For SFT training, padding should be on the right (if overflow occurs)
    tokenizer.padding_side = 'right'

    # processor
    processor = AutoProcessor.from_pretrained(vla_args.processor_name_or_path, num_crops=1, trust_remote_code=True)
    
    # use float16 (V100 does not support bfloat16)
    torch_dtype = torch.float16

    model_kwargs = dict(
        ignore_mismatched_sizes=True, # brutal force
        _attn_implementation='eager',
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        use_cache=False
    )

    # Initialize LLM
    model_vla = AutoModelForCausalLM.from_pretrained(vla_args.model_name_or_path, **model_kwargs)
    # model_vla.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128) # pad to multiple of 128 to improve performance
    model_vla = model_vla.eval().to(device)


    # 1. encode the images and actions
    # the src_filepath should contain the following fields
    # task_description, image_paths
    # an example
    data_args.src_filepath = "/mnt/data-rundong/robot_datasets/tokenizer-training/pizza_preprocessed_for_pie/test/test.jsonl"
    with open(data_args.src_filepath, 'r') as f:
        lines = f.readlines()
        line = lines[0]
        instance_data = json.loads(line)

    # call the models, override original actions and clip description with the predicted ones
    instance_data = call_models(instance_data, processor, tokenizer, model_vla, device)

    print(instance_data['clip_description'])
    print(instance_data['actions'])

    # call the robot, override the image_paths and actions with the actual ones
    # instance_data = call_robot(instance_data, robot)

if __name__ == '__main__':
    main()



