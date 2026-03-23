# Load the Model 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REASONING_SINGLE_PROMPT_TEMPLATE = """
[客户问题]
{question}

[客服A的回答开始]
{answer_a}
[客服A的回答结束]
[客服B的回答开始]
{answer_b}
[客服B的回答结束]
请将最终判定结果严格按照以下格式输出：\n若 客服A 更好，则输出：<answer>[[A]]</answer>\n若 客服B 更好，则输出：<answer>[[B]]</answer>
"""

model_name = "/root/autodl-tmp/meta_v2/RM-R1-Dpsk-Distilled-7B_v2-LR1.0e-6/global_step_93/actor/merged_model"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto" # or specify the specific device map if needed
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#"winner": "model_a",
# Single Turn Example - from Reward Bench 
prompt = "我昨天下单的空调预约今天14:00-16:00安装，师傅爽约还不接电话，我请了半天假白等了。这样的服务谁负责？需要我重新预约吗？"
answer_a = "真抱歉让您白等。我这边马上联系安装商催派，并为您改约今晚或明早的优先时段，确认后第一时间回电；额外补偿30元券。请留方便接听的号码和时段。若再次爽约可申请延误赔付。" # Accepted
answer_b = "非常抱歉给您带来不便，您的反馈我们已记录并会持续优化服务。安装以系统排期为准，请您耐心等待师傅联系，届时会以短信或电话通知。给您造成困扰再次抱歉，感谢理解与支持。如需改约可在订单页自助操作。"  # Rejected

user_prompt_single = REASONING_SINGLE_PROMPT_TEMPLATE.format(
    question=prompt,
    answer_a=answer_a,
    answer_b=answer_b
) 

conversation = [
    {"role":"user", "content": user_prompt_single}
]

input_ids = tokenizer.apply_chat_template(
    conversation, 
    tokenize=True, 
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

generation = model.generate(
    input_ids=input_ids,
    max_new_tokens=8192, # For optimal benchmarking, set this to unlimited (e.g., 50000)
    do_sample=False,
)

completion = tokenizer.decode(
    generation[0][len(input_ids[0]):], 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=True
)

print(completion)