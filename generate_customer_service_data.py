"""
生成AI客服训练数据脚本
目标：生成3000条客服对话数据，用于训练Reward Model
要求：数据多样化，评估AI感弱、温馨、不易引发投诉的回复
"""

import json
import random
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from llm import call_llm

# 数据生成提示词
DATA_GENERATION_PROMPT = """你是一个专业的数据标注专家，负责为AI客服训练生成高质量的对比数据。

任务：生成一条客服场景的对话数据，包含：
1. 一个客户问题
2. 两个客服回答（A和B）
3. 标注哪个回答更好

要求：
- 客户问题要真实、具体，涵盖电商客服的常见场景
- 客服A应该是机械化、AI感强、可能引发不满的回答
- 客服B应该是温馨、有人情味、主动解决问题的回答
- **重要：两个回答的长度必须相近（差距不超过20%），避免长度成为判断标准**
- 回答长度在50-150字之间，可以有变化
- 好的回答不一定要更长，关键是语气、态度和解决问题的方式
- 每次生成的场景要有多样性
- 不要思考，直接输出结果

场景类别（随机选择一个）：
{category}

长度控制策略（随机选择一种）：
{length_strategy}

请严格按照以下JSON格式输出，不要有任何额外说明：
{{
  "customer_question": "客户的具体问题",
  "answer_a": "客服A的机械化回答",
  "answer_b": "客服B的温馨回答",
  "winner": "model_b"
}}

注意：
1. 只输出JSON，不要有其他文字
2. winner必须是"model_a"或"model_b"（通常应该是model_b）
3. 回答要符合中国电商客服的语言习惯
4. 直接输出，不要思考过程
5. **确保两个回答长度相近，不要让好的回答总是更长**
"""

# 长度控制策略
LENGTH_STRATEGIES = [
    "两个回答长度相近，都是简短回答（50-80字）",
    "两个回答长度相近，都是中等长度（80-120字）",
    "两个回答长度相近，都是较长回答（120-150字）",
    "差的回答稍长（啰嗦、重复），好的回答简洁有力",
    "好的回答稍长（提供更多帮助），但不超过差的回答20%",
]

# 客服场景类别
CATEGORIES = [
    "物流延迟 - 客户着急收货",
    "商品质量问题 - 客户收到瑕疵品",
    "退换货咨询 - 客户想要退货或换货",
    "优惠活动咨询 - 客户询问折扣和优惠券",
    "售后服务 - 客户要求维修或退款",
    "商品信息咨询 - 客户询问商品详情",
    "支付问题 - 客户支付遇到障碍",
    "账户问题 - 客户账户异常或被冻结",
    "发票开具 - 客户需要开具发票",
    "配送地址变更 - 客户要修改收货地址",
    "价格争议 - 客户对价格有疑问",
    "商品推荐 - 客户寻求购买建议",
    "客户投诉 - 客户对服务不满",
    "订单取消 - 客户想要取消订单",
    "会员权益 - 客户咨询会员福利",
]

def generate_one_data_item(model_name: str = "gpt-5") -> Dict:
    """生成一条数据"""
    category = random.choice(CATEGORIES)
    length_strategy = random.choice(LENGTH_STRATEGIES)
    prompt = DATA_GENERATION_PROMPT.format(category=category, length_strategy=length_strategy)

    try:
        response = call_llm(
            model_name=model_name,
            prompt=prompt,
            stream=False,
            show_thinking=False
        )

        # 清理响应，提取JSON部分
        response = response.strip()

        # 尝试找到JSON部分
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        # 解析JSON
        data = json.loads(response)

        # 验证必要字段
        required_fields = ["customer_question", "answer_a", "answer_b", "winner"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"缺少字段: {field}")

        # 随机交换A和B的位置（50%概率）
        if random.random() < 0.5:
            # 交换answer_a和answer_b
            data['answer_a'], data['answer_b'] = data['answer_b'], data['answer_a']
            # 交换winner标签
            data['winner'] = 'model_a' if data['winner'] == 'model_b' else 'model_b'

        # 构造最终格式
        formatted_data = {
            "context_messages": [{
                "role": "user",
                "content": f"[客户问题]\n{data['customer_question']}\n\n"
                          f"[客服A的回答开始]\n{data['answer_a']}\n[客服A的回答结束]\n\n"
                          f"[客服B的回答开始]\n{data['answer_b']}\n[客服B的回答结束]"
            }],
            "winner": data["winner"]
        }

        return formatted_data

    except Exception as e:
        print(f"生成数据时出错: {e}")
        print(f"原始响应: {response if 'response' in locals() else 'N/A'}")
        return None

def generate_dataset(
    total_count: int = 3000,
    save_interval: int = 100,
    output_file: str = "customer_service_dataset.jsonl",
    model_name: str = "gpt-5",
    num_workers: int = 6
):
    """
    生成完整数据集（多线程版本）

    Args:
        total_count: 总数据量
        save_interval: 每隔多少条保存一次
        output_file: 输出文件路径
        model_name: 使用的模型名称
        num_workers: 线程数量
    """
    dataset = []
    lock = threading.Lock()  # 线程锁，保护共享数据
    failed_count = 0

    print(f"开始生成{total_count}条数据，使用模型: {model_name}")
    print(f"使用 {num_workers} 个线程并发生成")
    print(f"每{save_interval}条保存一次到 {output_file}\n")

    def generate_and_collect(task_id):
        """线程任务：生成一条数据"""
        nonlocal failed_count
        data_item = generate_one_data_item(model_name)

        with lock:
            if data_item:
                dataset.append(data_item)
                # 打印生成的数据完整内容
                content = data_item['context_messages'][0]['content']
                tqdm.write(f"\n{'='*80}")
                tqdm.write(f"[第 {len(dataset)} 条数据] Winner: {data_item['winner']}")
                tqdm.write(f"{'='*80}")
                tqdm.write(content)
                tqdm.write(f"{'='*80}\n")
                return True
            else:
                failed_count += 1
                tqdm.write(f"✗ 任务 {task_id} 生成失败，跳过")
                return False

    # 使用线程池并发生成
    with tqdm(total=total_count, desc="生成数据", unit="条") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(generate_and_collect, i) for i in range(total_count)]

            # 处理完成的任务
            completed = 0
            for future in as_completed(futures):
                completed += 1
                pbar.update(1)
                pbar.set_postfix({"成功": len(dataset), "失败": failed_count})

                # 每隔save_interval条保存一次（全量保存）
                if completed % save_interval == 0 or completed == total_count:
                    with lock:
                        tqdm.write(f"\n💾 保存数据到 {output_file} (当前共 {len(dataset)} 条)...")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            for item in dataset:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        tqdm.write(f"✓ 已保存 {len(dataset)} 条数据\n")

    print(f"\n✓ 数据生成完成！总共生成 {len(dataset)} 条有效数据")
    print(f"数据已保存到: {output_file}")

def preview_data(output_file: str = "customer_service_dataset.jsonl", num_samples: int = 3):
    """预览生成的数据"""
    print(f"\n预览 {output_file} 的前 {num_samples} 条数据：\n")

    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            print(f"=== 数据 {i+1} ===")
            print(f"Winner: {data['winner']}")
            print(f"内容预览: {data['context_messages'][0]['content'][:200]}...")
            print()

if __name__ == "__main__":
    # 生成数据集
    generate_dataset(
        total_count=3000,
        save_interval=100,
        output_file="customer_service_dataset.jsonl",
        model_name="gpt-5",
        num_workers=6  # 使用6个线程并发
    )

    # 预览数据
    preview_data("customer_service_dataset.jsonl", num_samples=3)
