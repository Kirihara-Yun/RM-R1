"""
自定义的 RayPPOTrainer 子类，专门用于 RM 项目的 GRPO 训练。
主要作用：
1. 继承自 verl 框架的 `RayPPOTrainer` 基类，复用其核心训练循环逻辑。
2. 重写 `_create_dataloader` 方法，以使用我们自定义的数据集类 `RubricRMDataset`。
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf
from omegaconf import open_dict
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
# 从 verl 框架导入基类 `RayPPOTrainer`，这是一个通用的、功能完整的 PPO 训练器。
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as _RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn

# 导入我们自定义的数据集类，它专门为 GRPO 的排序偏好数据设计。
from rm_r1.verl.utils.dataset.rl_dataset import RubricRMDataset


# 把做好的数据集给model
class RubricRMRayPPOTrainer(_RayPPOTrainer):
    """
    RubricRMRayPPOTrainer 是 RayPPOTrainer 的子类。
    通过重写 `_create_dataloader` 方法，我们将默认的数据加载逻辑替换为使用我们自己的 `RubricRMDataset`。
    其他所有功能（如 init_workers, fit, _validate 等）都直接继承自父类，无需修改。
    """
    
    def _create_dataloader(self):
        """
        创建训练和验证用的 DataLoader。
        这个方法会在 `RayPPOTrainer` 的 `__init__` 中被调用。
        
        核心步骤：
        1. 实例化自定义的 `RubricRMDataset`，用于加载我们准备好的 JSONL 格式数据。
        2. 根据配置选择 Sampler（随机采样或顺序采样）。
        3. 创建支持断点续训的 `StatefulDataLoader`。
        4. 计算总训练步数，并注入到优化器配置中（用于学习率调度）。
        """
        
        # --- 1. 创建训练数据集 ---
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RubricRMDataset(
            files=self.config.data.train_files,           # 训练数据文件路径（JSONL 格式）
            tokenizer=self.tokenizer,                     # Tokenizer，用于将文本转换为 token IDs
            processor=self.processor,                     # Processor（如果是多模态模型则需要，否则为 None）
            prompt_key=self.config.data.prompt_key,       # 数据中 prompt 字段的键名（默认是 'messages'）
            image_key=self.config.data.get('image_key', 'images'), # 多模态数据中图像字段的键名
            max_prompt_length=self.config.data.max_prompt_length,  # Prompt 的最大 token 长度
            return_raw_chat=self.config.data.get('return_raw_chat', False), # 是否返回原始的对话数据
            truncation=self.config.data.truncation,       # 截断策略（'error' 表示遇到过长数据会报错）
            filter_overlong_prompts=self.config.data.filter_overlong_prompts, # 是否过滤掉超长的 prompt
        )
        
        # --- 2. 根据配置选择 Sampler ---
        # use sampler for better ckpt resume
        # Sampler 决定了数据的采样顺序。使用 Sampler 而不是直接 shuffle，是为了支持更好的断点续训。
        if self.config.data.shuffle:
            # 使用随机采样器，并设置固定的随机种子以保证可复现性
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(
                self.config.data.get('seed', 1),
            )
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator,
            )
        else:
            # 使用顺序采样器
            sampler = SequentialSampler(data_source=self.train_dataset)

        # --- 3. 创建训练 DataLoader ---
        # 使用 `StatefulDataLoader` 而不是标准的 `DataLoader`，是因为它能保存当前的采样状态。
        # 这样在训练中断后，可以从上次中断的位置继续，而不会重复采样之前的数据。
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size, # 批大小（对应 shell 脚本中的 TRAIN_BS）
            num_workers=8,                                # 数据加载的并行进程数
            drop_last=True,                               # 丢弃最后一个不完整的 batch
            collate_fn=collate_fn,                        # 自定义的数据整理函数，用于将多个样本组装成一个 batch
            sampler=sampler,
        )

        # --- 4. 创建验证数据集和 DataLoader ---
        self.val_dataset = RubricRMDataset(
            files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            max_prompt_length=self.config.data.max_prompt_length,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation=self.config.data.truncation,
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            # 验证数据集会作为一个完整的 batch 一次性送入推理引擎，由引擎自己调度显存。
            # 因此 batch_size 设置为整个验证集的大小。
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,                                # 验证时不需要打乱数据
            drop_last=False,                              # 验证时不丢弃最后的数据
            collate_fn=collate_fn,
        )

        # --- 5. 检查 DataLoader 的有效性 ---
        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader,
        ) == 1, 'Validation dataloader must have a single batch, which inference engines will schedule the memory themselves.'

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # --- 6. 计算总训练步数，并注入到优化器配置中 ---
        # inject total_training_steps to actor/critic optim_config. This is hacky.
        # 总步数 = 每个 epoch 的步数 × epoch 数量
        # 这个信息会被用于学习率调度器（如 warmup, cosine decay 等）。
        total_training_steps = len(
            self.train_dataloader,
        ) * self.config.trainer.total_epochs

        # 如果用户手动指定了总步数，则使用用户的配置
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        # 将总步数注入到 actor 和 critic 的优化器配置中
        # 注意：OmegaConf 的配置对象默认是只读的（struct=True），需要先用 `open_dict` 打开。
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
