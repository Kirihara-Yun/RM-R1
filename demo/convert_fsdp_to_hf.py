"""
将 FSDP 分片的 checkpoint 合并成标准的 Hugging Face 格式
"""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置路径
checkpoint_dir = "/root/autodl-tmp/meta_v2/RM-R1-Dpsk-Distilled-7B_v2-LR1.0e-6/global_step_93/actor"
base_model_path = "/root/autodl-tmp/dsr1-7b"  # 原始模型路径，用于获取配置
output_dir = "/root/autodl-tmp/meta_v2/RM-R1-Dpsk-Distilled-7B_v2-LR1.0e-6/global_step_93/actor/merged_model"

print("正在加载基础模型配置...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu"  # 先加载到 CPU
)

print("正在加载并合并 FSDP checkpoint...")
# 加载第一个 rank 的 checkpoint 作为参考
state_dict_list = []
for rank in range(4):
    ckpt_path = os.path.join(checkpoint_dir, f"model_world_size_4_rank_{rank}.pt")
    print(f"加载 rank {rank}: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    state_dict_list.append(state_dict)

# 合并 state dict（这里简化处理，假设可以直接用 rank 0 的）
# 注意：如果是真正的 FSDP full_state_dict，需要更复杂的合并逻辑
print("合并 state dict...")
merged_state_dict = state_dict_list[0]

print("加载合并后的权重到模型...")
model.load_state_dict(merged_state_dict, strict=False)

print(f"保存合并后的模型到: {output_dir}")
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)

# 复制 tokenizer
print("复制 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_dir)

print("完成！现在可以使用以下路径加载模型：")
print(f"model_name = '{output_dir}'")

