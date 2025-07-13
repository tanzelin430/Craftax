#!/usr/bin/env python3
"""
Craftax Dataset - 简单版本
只提供初始的环境观察作为 prompt
"""

import os

# 让JAX使用CPU，避免与训练GPU冲突
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.processing_utils import ProcessorMixin

# Agent Loop 会自己管理环境，Dataset 只提供初始 prompts


class CraftaxDataset(Dataset):
    """
    简单的 Craftax 数据集
    只负责生成初始环境观察作为 prompt
    """

    def __init__(
        self,
        data_files: List[str],  # 可以忽略，我们不从文件读取
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        config,  # OmegaConf config object
        num_episodes: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # 从 OmegaConf 配置中读取参数
        self.num_episodes = config.num_episodes

        # check
        if self.num_episodes is None:
            raise ValueError("num_episodes is not set")

        print(f"Dataset will provide {self.num_episodes} prompts for Agent Loops")

        # 生成初始观察
        self.data = self._generate_initial_observations()

    def _generate_initial_observations(self) -> List[Dict[str, Any]]:
        """生成初始环境观察"""
        print(f"Generating {self.num_episodes} initial Craftax prompts...")

        data = []
        for i in range(self.num_episodes):
            # 创建一个简单的初始 prompt，Agent Loop 会自己管理环境
            # 我们只需要告诉 Agent Loop 这是第几个 episode
            initial_prompt = f"Start a new Craftax episode. This is episode {i}."

            data.append(
                {
                    "prompt": [
                        {"role": "user", "content": initial_prompt, "episode_id": i}
                    ],
                    "data_source": "craftax",
                    "agent_name": "craftax_agent",  # 指定使用 CraftaxAgentLoop
                    "episode_id": i,
                    "extra_info": {
                        "index": i,
                        "tools_kwargs": {},
                        "interaction_kwargs": {},
                        "need_tools_kwargs": False,
                    },
                }
            )

        print(f"Generated {len(data)} initial prompts for Agent Loops")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """返回符合 Verl 框架要求的数据格式"""
        item = self.data[idx]

        # 创建固定长度的 dummy tokens，避免长度不一致问题
        # Agent Loop 会自己处理真正的环境交互，这里只是占位符
        dummy_length = 10  # 固定长度
        input_ids = torch.tensor([1] * dummy_length, dtype=torch.long)  # 全部设为 1
        attention_mask = torch.tensor([1] * dummy_length, dtype=torch.long)  # 全部有效
        position_ids = torch.arange(dummy_length, dtype=torch.long)

        # 返回符合 Verl 框架要求的格式
        return {
            # 必要的张量字段（固定长度，避免 stack 错误）
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            # 非张量字段（会被 collate_fn 转换为 object 数组）
            "raw_prompt": item["prompt"],  # 原始 prompt 消息
            "raw_prompt_ids": input_ids.tolist(),  # 添加这个字段，框架需要
            "data_source": item["data_source"],
            "agent_name": "craftax_agent",  # 使用框架支持的agent类型
            "episode_id": item["episode_id"],
            "index": idx,
            "tools_kwargs": item["extra_info"].get("tools_kwargs", {}),
            "interaction_kwargs": item["extra_info"].get("interaction_kwargs", {}),
            # 为 NaiveRewardManager 添加必要字段
            "reward_model": {
                "ground_truth": "craftax_environment_reward"  # 占位符，实际奖励从响应中提取
            },
        }


# if __name__ == "__main__":
# 测试数据集
# config = {'max_episode_steps': 50}
# dataset = CraftaxDataset([], None, None, config, num_episodes=5)

# print(f"Dataset size: {len(dataset)}")
# print(f"First prompt length: {len(dataset[0]['prompt'])}")
# print(f"First prompt preview: {dataset[0]['prompt'][:200]}...")
