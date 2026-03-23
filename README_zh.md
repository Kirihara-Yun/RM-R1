### RM-R1 训练流程
#### 一、环境配置
1. 创建并激活 conda 环境
```bash
conda create -n rm-r1-1 python=3.11 -y
conda activate rm-r1-1
```

2. 安装 veRL（固定提交版本）
```bash
git clone https://github.com/volcengine/verl && cd verl
git checkout e49fb572bf85a8f0ef7124c898f509bd6d9832a1
pip install -e . && cd ..
```

3. 安装 vLLM（固定提交+flash-attention 加速）
```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout ed6e9075d31e32c8548b480a47d1ffb77da1f54c
git config --global user.name "zyQin"
git config --global user.email "Kiriharayun@163.com"
git cherry-pick caac5c2e597b1780c3df54a537c34e6061c32cff
export VLLM_COMMIT=ed6e9075d31e32c8548b480a47d1ffb77da1f54c
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
VLLM_USE_PRECOMPILED=1 pip install --editable .
pip install flash-attn==2.7.2.post1 --no-build-isolation && cd ..
```

#### 二、训练执行
跳过 SFT 直接执行 GRPO 训练（以 DeepSeek 蒸馏 7B 为例）：
```bash
cd /root/autodl-tmp/RM-R1
bash ./rm_r1/scripts/RLVR/local/train_rm_r1_rlvr_dpsk_distilled_7b.sh
```

### 总结
1. 核心依赖：需安装指定提交版本的 veRL 和 vLLM，其中 vLLM 需集成 flash-attention 2 实现 2 倍以上加速；
2. 训练方式：RM-R1 无需先做 SFT，可直接基于推理模型执行 GRPO 训练；
3. 执行入口：训练脚本位于 `rm_r1/scripts/RLVR/local/` 目录，示例为 DeepSeek 蒸馏 7B 模型的训练脚本。