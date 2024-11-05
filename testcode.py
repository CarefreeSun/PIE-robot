from llm_backbone import MistralInVisionActionFeatMask, Codebook
import torch
import os
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import MistralConfig, MistralForCausalLM

tokenzier = AutoTokenizer.from_pretrained('../mistral-7B-v0.1')
va_embed = Codebook(16384, 256)
model_kwargs = dict(
    # revision=model_args.model_revision,
    use_flash_attention_2=False,
    torch_dtype=torch.float16,
    # trust_remote_code=True,
    use_cache=False
)

model = MistralInVisionActionFeatMask.from_pretrained('../checkpoint-32000', tokenzier, va_embed, 0.0, **model_kwargs)
for name, param in model.named_parameters():
    print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
    if 'layers' in name:
        name_split = name.split('.')
        layer_id = int(name_split[2])
        if layer_id <= 27:
            param.requires_grad = False

for name, param in model.named_parameters():
    print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")



# def load_safetensors_weights(checkpoint_dir): 
#     weights_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')] 
#     for weights_file in weights_files: 
#         weights_path = os.path.join(checkpoint_dir, weights_file) 
#         with safe_open(weights_path, framework="pt", device="cpu") as f: 
#             for key in f.keys(): 
#                 print(key, f.get_tensor(key).shape)
#     #             model.state_dict()[key].copy_(f.get_tensor(key)) 
#     # return model
# load_safetensors_weights('../model_weights')
# key = 'model.embeddings.json'
# print(key.split('model.')[-1])

# model = MistralInVisionActionFeatMask.from_pretrained('asas')
# model.generate()