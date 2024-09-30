import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
# from thop import profile
# from thop import clever_format
# flops, params = profile(net, inputs=(x, ))
# # print(flops, params)
# macs, params = clever_format([flops, params], "%.3f")

path_to_model = "/root/OneBitQuantizer/huggingface/model/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# model = AutoModel.from_pretrained(path_to_model).to(torch.float16).to("cuda")
model = LlamaForCausalLM.from_pretrained(path_to_model).to(torch.float16).to("cuda")


before_replace_mem = torch.cuda.memory_allocated(0)/1024/1024/1024

import onebit_sparse_mul


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

total_params = 0
for idx, (n, m) in enumerate(model.named_modules()):
    if isinstance(m, nn.Linear):
        print(idx, '->', m, m.weight.shape, m.weight.dtype, m.weight.numel())
        in_feat = m.weight.shape[1]
        out_feat = m.weight.shape[0]
        linear = onebit_sparse_mul.BinaryLinear_2_4(in_feat, out_feat).to("cuda")
        linear.pack(m.weight)
        # total_params += in_feat*out_feat
        total_params += m.weight.numel()
        if getattr(m, "bias", None) != None:
            linear.bias = m.bias
            total_params += m.bias.numel()
            # total_params += m.bais..numel()
        _set_module(model, n, linear)
        
after_replace_mem = torch.cuda.memory_allocated(0)/1024/1024/1024
print("before model:{} GB".format(before_replace_mem))
print("after model:{} GB".format(after_replace_mem))
print(f"replaced total params {total_params}")

fp16_params_memory = total_params * 16 / 8 /1024/1024/1024
sparse_1bit_params_memory = total_params *1.5 / 8 /1024/1024/1024
print("In our theory, before replaced total params cost:", fp16_params_memory, "GB;", 
      f"rest params mem{before_replace_mem-fp16_params_memory}GB")
print("In our theory, after replaced total params cost:", sparse_1bit_params_memory, "GB;"
      f"rest params mem{after_replace_mem-sparse_1bit_params_memory}GB")
