#!/bin/bash
# =================================================================
# === 环境变量设置 ===
# =================================================================
# -- 基础环境配置 --
export VLLM_ATTENTION_BACKEND=XFORMERS # 指定 vLLM 使用 xFormers 作为注意力计算后端，以提升性能。
export VLLM_USE_V1=0                  # vLLM 相关配置，通常用于版本兼容或特性开关。
export VERL_PPO_LOGGING_LEVEL="INFO"  # 设置 GRPO训练过程的日志级别为 INFO，方便观察过程。
export WANDB_DISABLED=true            # 禁用 Weights & Biases 日志记录，避免需要登录或上传。

# -- GPU 配置 --
N_GPU=1                               # 设置用于训练的 GPU 数量。

# =================================================================
# === 模型与路径设置 ===
# =================================================================
# -- 基础模型配置 --
MODEL_PATH=/root/autodl-tmp/dsr1-7b   # 指定要进行微调的基础模型（Policy Model）所在的路径。

# =================================================================
# === 核心训练参数 ===
# =================================================================
# -- 优化器与学习率 --
LR=1.0e-6                             # 设置学习率（Learning Rate），这是决定模型更新步长的关键参数。

# -- 显存与批大小 --
GPU_MEM_UTILIZATION=0.5               # vLLM 引擎的显存占用率。如果遇到 OOM (Out of Memory)，可以调低此值。
TOTAL_EPISODES=1                      # 训练的总轮次（Epochs）。
SAVE_EVERY_STEP=100                   # 每隔多少个 step 保存一次模型 checkpoint。
TEST_EVERY_STEP=100000                # 每隔多少个 step 在验证集上进行一次评估（当前值很大，等于不评估）。
TRAIN_BS=32                           # [核心] Rollout 批大小。决定了每个 step 从数据集中取多少条 prompt 用于生成。
PPO_MINI_BS=8                         # [核心] PPO 训练批大小。在 Rollout 之后，用于模型参数更新的批次大小。

# -- 学习率预热 --
WARMUP_STYLE=constant                 # 学习率预热策略，'constant' 表示不使用预热。

# -- Token 长度控制 --
MAX_PROMPT_LENGTH=4096                # Prompt 的最大长度（tokens）。如果遇到 OOM，可以调低。
MAX_RESPONSE_LENGTH=8192              # 生成 Response 的最大长度（tokens）。如果遇到 OOM，可以调低。

# -- 分布式训练配置 --
TRAIN_PER_GPU=4                       # 每个 GPU 上进行 PPO 更新时的真实批大小。必须能被 PPO_MINI_BS 整除。
FORWARD_PER_GPU=4                     # 在计算 logprob 等前向传播任务时，每个 GPU 的批大小。必须能被 TRAIN_BS 整除。

# =================================================================
# === 日志与保存设置 ===
# =================================================================
PROJECT_NAME=RM-R1
EXPERIMENT_NAME=RM-R1-Dpsk-Distilled-7B_v2-LR${LR} # 实验名称，会用作日志和保存目录的一部分。
SAVE_NAME=RM-R1-Dpsk-Distilled-7B_v2-LR${LR}       # 保存 checkpoint 的目录名。
SAVE_META_DIR="/root/autodl-tmp/meta_v2"            # 所有实验结果、日志和模型的根目录。

# =================================================================
# === 奖励函数设置 ===
# =================================================================
# -- 指定自定义奖励函数 --
REWARD_PATH=./rm_r1/verl/utils/reward_score/lm_as_judge.py  # 指向实现了奖励计算逻辑的 Python 文件。
REWARD_FUNC_NAME=lm_as_judge_match                           # 上述文件中要调用的具体函数名。

# =================================================================
# === 数据集设置 ===
# =================================================================
TRAIN_TASK="/root/autodl-tmp/RM-R1/rm_r1/dataset/mix_data/train.jsonl"  # 训练数据集的路径。
EVAL_TASK="/root/autodl-tmp/RM-R1/rm_r1/dataset/mix_data/train.jsonl"   # 验证数据集的路径。

# =================================================================
# === 固定设置 (通常无需修改) ===
# =================================================================
MAX_NUM_BATCHED_TOKENS=$(($MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH)) # vLLM 需要的参数，表示一个批次中能处理的最大 token 数。

# =================================================================
# === Ray 分布式环境启动 ===
# =================================================================
# -- 清理并重启 Ray --
ray stop                              # 停止任何可能已在运行的 Ray 实例。
sleep 5                               # 等待 5 秒确保进程已关闭。
ray stop                              # 再次尝试停止，确保环境干净。

# -- 启动新的 Ray 集群 --
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${N_GPU} # 启动一个 Ray head 节点，并指定使用 ${N_GPU} 张 GPU。

# =================================================================
# === Python 训练主程序调用 ===
# =================================================================
# -- 核心启动命令 --
# 注意：以下所有参数都是传给 Python 程序的，通过 Hydra 库进行解析
python3 -m rm_r1.verl.trainer.main_ppo \
    data.train_files=${TRAIN_TASK} \
    data.val_files=${EVAL_TASK} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_BS} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${TRAIN_PER_GPU} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_PER_GPU} \
    custom_reward_function.path=${REWARD_PATH} \
    custom_reward_function.name=${REWARD_FUNC_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.logger='["console"]' \
    trainer.total_epochs=${TOTAL_EPISODES} \
    trainer.save_freq=${SAVE_EVERY_STEP} \
    trainer.test_freq=-1 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.warmup_style=${WARMUP_STYLE} \
    trainer.default_local_dir=${SAVE_META_DIR}/${SAVE_NAME}

# =================================================================
# === 训练结束后清理 ===
# =================================================================
ray stop                              # 训练完成后，停止 Ray 集群，释放资源。