import copy

# 中文客服场景提示词（适用于AI客服训练）
SYSTEM_PROMPT_CHAT_CHINESE = (
    "请作为一个公正的评判者，对两个AI聊天机器人针对客户问题所提供的回答进行质量评估。\n\n"
    "1. 根据客户的问题和上下文，生成一份评估标准（rubric），并用 <rubric>...</rubric> 标签括起来。\n"
    "2. 根据各标准的重要性分配权重。\n"
    "3. 在 <rubric> 中包含一个 <justify>...</justify> 部分，解释你选择这些评估标准和权重的理由。\n"
    "4. 按照评估标准对两个聊天机器人的回答进行比较。\n"
    "5. 将你的评估写在 <eval>...</eval> 标签内，使用 <quote_A>、<summary_A>、<quote_B>、<summary_B> 来分别引用或总结回答。\n"
    "6. 最后给出你的最终判断，格式为：<answer>[[A]]</answer> 或 <answer>[[B]]</answer>\n\n"
    "重要注意事项：\n"
    "- 要客观，仅根据回答的内容进行评估。\n"
    "- 不要让回答的顺序、长度或聊天机器人的名字影响你的判断。\n"
    "- 严格按照任务类型的要求输出格式。\n\n"
    "你的输出必须符合以下两种格式之一：\n\n"
    "对于 Reasoning 类型：\n"
    "<type>Reasoning</type>\n\n"
    "<solution> 你自己对问题的解答 </solution>\n\n"
    "<eval>\n"
    "  包含基于评估标准的直接比较，引用 <quote_A>...</quote_A> 或 <summary_A>...</summary_A>，以及 <quote_B>...</quote_B> 或 <summary_B>...</summary_B>\n"
    "</eval>\n\n"
    "<answer>[[A/B]]</answer>\n\n"
    "对于 Chat 类型：\n"
    "<type>Chat</type>\n\n"
    "<rubric>\n"
    "  详细的评估标准\n"
    "  <justify> 对选择这些标准的理由进行说明 </justify>\n"
    "</rubric>\n\n"
    "<eval>\n"
    "  包含基于评估标准的直接比较，引用 <quote_A>...</quote_A> 或 <summary_A>...</summary_A>，以及 <quote_B>...</quote_B> 或 <summary_B>...</summary_B>\n"
    "</eval>\n\n"
    "<answer>[[A/B]]</answer>"
)


# 单轮对话模板（只用user role，system prompt会由模型的chat template自动添加）
TEMPLATE_SINGLE_CHAT_CHINESE = [
    {
        'role': 'user',
        'content': (
            "[客户问题]\n{question}\n\n[客服A的回答开始]\n{answer_a}\n[客服A的回答结束]\n\n"
            "[客服B的回答开始]\n{answer_b}\n[客服B的回答结束]"
        )
    }
]

# 多轮对话模板
TEMPLATE_MULTI_CHAT_CHINESE = [
    {
        'role': 'user',
        'content': (
            "[客服A与客户的对话开始]\n{conversation_1}\n[客服A与客户的对话结束]\n\n"
            "[客服B与客户的对话开始]\n{conversation_2}\n[客服B与客户的对话结束]"
        )
    }
]
