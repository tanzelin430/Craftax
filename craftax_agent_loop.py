#!/usr/bin/env python3
"""
Craftax Agent Loop for Verl
管理完整的 Craftax 环境交互过程
"""

import asyncio
import time
import numpy as np
from typing import Any, Dict, List
from uuid import uuid4

from transformers import AutoTokenizer
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from llm_agent_wrapper import CraftaxLLMWrapper


class CraftaxAgentLoop(AgentLoopBase):
    """
    Craftax Agent Loop
    使用类级别环境持久化，因为 Verl 框架会为每次请求创建新的实例
    
    设计思路：
    - Verl 为每次请求创建新的 Agent Loop 实例
    - 我们在类级别存储环境状态，实现跨实例的环境持久化
    - 每个 worker/session 对应一个持久的环境
    """
    
    _max_episode_steps = 50  # 类级别配置
    _environments = {}  # 类级别环境存储: worker_id -> {env_wrapper, current_state, current_obs, episode_step_count, episode_id}
    
    def __init__(self, config: DictConfig, server_manager, tokenizer: AutoTokenizer):
        super().__init__(config, server_manager, tokenizer)
        
        # 从配置中获取序列长度限制
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        
        # 环境现在通过 messages 中的 episode_id 来标识，不需要实例级别的标识
    
    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """初始化类共享配置并预创建所有环境"""
        if cls._class_initialized:
            return
        
        cls._class_initialized = True
        # 从配置中读取最大步数，默认 50
        cls._max_episode_steps = getattr(config, 'max_episode_steps', 50)
        
        # 获取环境数量（对应 train_batch_size）
        num_episodes = getattr(config.data, 'num_episodes', 16)
        
        # 获取当前进程 ID 作为 worker 标识
        import os
        worker_pid = os.getpid()
        
        # 预创建所有环境（每个 worker 创建相同的环境 ID，但在不同的进程空间中）
        print(f"🚀 Worker {worker_pid}: Pre-creating {num_episodes} Craftax environments...")
        for i in range(num_episodes):
            env_id = f"env_{i}"
            cls._create_environment(env_id)
        
        print(f"✅ Worker {worker_pid}: CraftaxAgentLoop class initialized with {num_episodes} environments, max_episode_steps={cls._max_episode_steps}")
    
    @classmethod
    def _create_environment(cls, env_id: str):
        """创建并初始化一个新环境"""
        # 创建环境
        env_wrapper = CraftaxLLMWrapper(
            env_name='Craftax-Symbolic-v1',
            max_episode_steps=cls._max_episode_steps
        )
        
        # 初始化环境状态
        seed = np.random.randint(0, 100000)
        current_obs, current_state = env_wrapper.reset(seed=seed)
        
        cls._environments[env_id] = {
            'env_wrapper': env_wrapper,
            'current_state': current_state,
            'current_obs': current_obs,
            'episode_step_count': 0,
            'episode_id': f"ep_{env_id}_{seed}"
        }
        
        print(f"🎮 Created environment {env_id} with seed {seed}")
    
    @classmethod
    def _get_environment(cls, env_id: str):
        """获取指定的预创建环境"""
        if env_id not in cls._environments:
            print(f"⚠️ Environment {env_id} not found in pre-created environments, creating on-demand")
            cls._create_environment(env_id)
            
        return cls._environments[env_id]
    
    @classmethod
    def _reset_environment(cls, env_id: str):
        """重置指定环境到新 episode"""
        env_data = cls._get_environment(env_id)
        
        # 重置环境
        seed = np.random.randint(0, 100000)
        current_obs, current_state = env_data['env_wrapper'].reset(seed=seed)
        
        # 更新状态
        env_data['current_state'] = current_state
        env_data['current_obs'] = current_obs
        env_data['episode_step_count'] = 0
        env_data['episode_id'] = f"ep_{env_id}_{seed}"
        
        print(f"🔄 Reset environment {env_id} with seed {seed}")
    
    async def run(self, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any]) -> AgentLoopOutput:
        """
        运行单步 Craftax 环境交互
        
        工作流程：
        1. 从 messages 中提取环境标识符
        2. 获取对应的持久化环境
        3. 使用当前环境观察生成 LLM 响应
        4. 解析动作并执行环境 step
        5. 更新持久化的环境状态
        
        Args:
            messages: 来自 dataset 的消息，包含环境标识信息
            sampling_params: LLM 采样参数
            
        Returns:
            AgentLoopOutput: 单步交互数据
        """
        start_time = time.time()
        request_id = uuid4().hex
        
        # 1. 从 messages 中提取环境标识符
        env_id = "default_env"
        if messages and len(messages) > 0:
            initial_message = messages[0]
            if isinstance(initial_message, dict):
                # 使用 episode_id 作为环境标识，确保不同 episode 有不同环境
                env_id = f"env_{initial_message.get('episode_id', 'default')}"
        
        # 2. 获取对应的持久化环境
        env_data = self._get_environment(env_id)
        
        # 3. 检查是否需要开始新 episode（当前观察为 None 表示上个 episode 结束了）
        if env_data['current_obs'] is None:
            self._reset_environment(env_id)
            env_data = self._get_environment(env_id)  # 重新获取更新后的数据
        
        # 3. 获取当前环境观察
        wrapped_obs = env_data['env_wrapper'].wrap_observation(env_data['current_obs'])
        
        # 4. 生成 LLM 响应
        prompt_list = [{"role": "user", "content": wrapped_obs}]
        formatted_text = self.tokenizer.apply_chat_template(
            prompt_list,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 训练效率考虑，禁用思考模式
        )
        
        prompt_ids = self.tokenizer.encode(formatted_text, add_special_tokens=True)
        
        # LLM 生成响应
        response_ids = await self.server_manager.generate(
            request_id=f"{request_id}_{env_data['episode_id']}_step{env_data['episode_step_count']}",
            prompt_ids=prompt_ids,
            sampling_params=sampling_params
        )
        
        # 解码响应
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # 5. 执行环境交互
        try:
            action_id = env_data['env_wrapper'].parse_llm_response(response_text)
        except Exception as e:
            print(f"⚠️ Action parsing failed: {e}, using default DO action")
            action_id = 5  # 默认 DO 动作
        
        # 执行动作并更新持久化的环境状态
        new_obs, new_state, reward, done, _ = env_data['env_wrapper'].step(env_data['current_state'], action_id)
        
        # 更新持久化的环境状态
        env_data['current_obs'] = new_obs
        env_data['current_state'] = new_state
        env_data['episode_step_count'] += 1
        
        # 检查是否需要重置环境
        if done:
            print(f"🏁 Episode {env_data['episode_id']} finished after {env_data['episode_step_count']} steps")
            # 环境结束，标记需要重置（但不销毁环境）
            env_data['current_obs'] = None
            env_data['current_state'] = None
        
        # 5. 在 response 后面加上当前步骤的奖励信息
        reward_info = f" [Reward: {reward:.3f}]"
        reward_tokens = self.tokenizer.encode(reward_info, add_special_tokens=False)
        response_ids.extend(reward_tokens)
        
        # 创建响应掩码
        response_mask = [1] * len(response_ids)
        
        # 6. 构建最终输出
        # 使用配置中的序列长度限制
        final_prompt_ids = prompt_ids[:self.prompt_length]
        final_response_ids = response_ids[:self.response_length]
        final_response_mask = response_mask[:len(final_response_ids)]
        
        # 构建指标
        total_time = time.time() - start_time
        metrics = {
            "generate_sequences": total_time * 0.8,  # 大部分时间在生成
            "tool_calls": total_time * 0.2,  # 少部分时间在环境交互
            "step_reward": reward,  # 当前步奖励
            "episode_step": env_data['episode_step_count'],  # 当前 episode 步数
            "done": done,  # 是否结束
            "action_id": action_id,  # 执行的动作
            "episode_id": str(env_data['episode_id']),  # episode 标识
            "env_id": env_id,  # 环境标识
        }
        
        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids,
            response_mask=final_response_mask,
            num_turns=1,  # 每次只执行一步
            metrics=AgentLoopMetrics(**metrics),
        )