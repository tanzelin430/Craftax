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

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AgentLoopMetrics,
)
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

    _max_episode_steps = None  # 类级别配置，从config动态获取
    _environments = (
        {}
    )  # 类级别环境存储: worker_id -> {env_wrapper, current_state, current_obs, episode_step_count, episode_id}
    _worker_counter = 0  # 原子计数器
    _worker_lock = None  # 线程锁
    _assigned_worker_id = None  # 当前worker分配到的ID
    _envs_per_worker = 4  # 每个worker管理的环境数量，默认值

    # Wandb reward tracking - 类级别共享
    _episode_cumulative_rewards = {}  # episode_id -> cumulative_reward
    _max_craftax_reward = 226.0  # Craftax最大奖励值
    _num_envs = 0  # 从config获取并缓存
    _rollout_n = 0  # 从config获取并缓存

    def __init__(self, trainer_config, server_manager, tokenizer: AutoTokenizer):
        # 直接调用父类的__init__，父类会处理trainer_config.config的访问
        super().__init__(trainer_config, server_manager, tokenizer)
        self.trainer_config = trainer_config  # 存储trainer配置

        # 从配置中获取序列长度限制
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        # 环境现在通过 messages 中的 episode_id 来标识，不需要实例级别的标识
        self.current_global_steps = 0  # 存储当前rollout的global_steps

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """初始化类共享配置并预创建当前worker负责的环境"""
        if cls._class_initialized:
            return

        cls._class_initialized = True
        # 从配置中读取最大步数，默认 100
        cls._max_episode_steps = getattr(config, "max_episode_steps", 100)

        # 获取配置信息并缓存为类变量
        num_episodes = getattr(config.data, "num_episodes", 32)  # 总环境数
        num_workers = getattr(
            config.actor_rollout_ref.rollout.agent, "num_workers", 32
        )  # worker数量

        # 缓存Wandb相关的固定配置
        cls._num_envs = num_episodes
        cls._rollout_n = getattr(config.actor_rollout_ref.rollout, "n")
        # initialize info
        print(
            f"Agent Initialized with num_episodes: {num_episodes}, num_workers: {num_workers}, rollout_n: {cls._rollout_n}"
        )
        # 获取当前进程信息
        import os

        worker_pid = os.getpid()

        # 简化方案：每个AgentLoop实例管理固定数量的环境
        # 因为在不同进程中，环境ID可以重复，由框架负责路由
        cls._envs_per_worker = num_episodes // num_workers  # 动态计算：128 // 64 = 2

        # 强制输出，避免被Ray聚合
        import sys

        sys.stdout.flush()
        sys.stderr.flush()

        print(f"🚀🚀🚀 CRAFTAX_WORKER_INIT_PID_{worker_pid} 🚀🚀🚀", flush=True)
        print(f"Creating {cls._envs_per_worker} environments", flush=True)

        # 每个实例创建固定的环境：env_0, env_1, ...
        for i in range(cls._envs_per_worker):
            env_id = f"env_{i}"
            cls._create_environment(env_id)

        print(
            f"✅ AgentLoop (PID {worker_pid}): Initialized {cls._envs_per_worker} environments, max_episode_steps={cls._max_episode_steps}"
        )

    def _get_current_global_step(self) -> int:
        """获取当前rollout的global_steps"""
        print(
            f"🔍 Current global_steps: {self.current_global_steps}, rollout_n: {self.__class__._rollout_n}"
        )
        return self.current_global_steps

    @classmethod
    def _create_environment(cls, env_id: str):
        """创建并初始化一个新环境"""
        # 创建环境
        env_wrapper = CraftaxLLMWrapper(
            env_name="Craftax-Symbolic-v1", max_episode_steps=cls._max_episode_steps
        )

        # 初始化环境状态
        seed = np.random.randint(0, 100000)
        current_obs, current_state = env_wrapper.reset(seed=seed)

        cls._environments[env_id] = {
            "env_wrapper": env_wrapper,
            "current_state": current_state,
            "current_obs": current_obs,
            "episode_step_count": 0,
            "episode_id": f"ep_{env_id}_{seed}",
        }

        print(f"🎮 Created environment {env_id} with seed {seed}")

    @classmethod
    def _get_environment(cls, env_id: str):
        """获取指定的预创建环境"""
        if env_id not in cls._environments:
            raise ValueError(
                f"❌ Environment {env_id} not found in worker's assigned environments. "
                f"Available environments: {list(cls._environments.keys())}"
            )

        return cls._environments[env_id]

    @classmethod
    def _reset_environment(cls, env_id: str):
        """重置指定环境到新 episode"""
        env_data = cls._get_environment(env_id)

        # 重置环境
        seed = np.random.randint(0, 100000)
        current_obs, current_state = env_data["env_wrapper"].reset(seed=seed)

        # 更新状态
        env_data["current_state"] = current_state
        env_data["current_obs"] = current_obs
        env_data["episode_step_count"] = 0
        env_data["episode_id"] = f"ep_{env_id}_{seed}"

        print(f"🔄 Reset environment {env_id} with seed {seed}")

    async def run(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        trajectory: Dict[str, Any] = None,
    ) -> AgentLoopOutput:
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
            trajectory: 包含global_steps等信息的轨迹字典
        """
        """
        Returns:
            AgentLoopOutput: 单步交互数据
        """
        # 从trajectory获取global_steps
        if trajectory and "step" in trajectory:
            self.current_global_steps = trajectory["step"]
        start_time = time.time()
        request_id = uuid4().hex

        # 1. 从 messages 中提取环境标识符并映射到本实例的环境
        episode_id = 0
        if messages and len(messages) > 0:
            initial_message = messages[0]
            if isinstance(initial_message, dict):
                episode_id = initial_message.get("episode_id", 0)

        # 将全局episode_id映射到当前实例的环境中的一个
        local_env_index = (
            episode_id % self._envs_per_worker
        )  # 动态映射到 0, 1, ..., envs_per_worker-1
        env_id = f"env_{local_env_index}"

        # 2. 获取对应的持久化环境
        env_data = self._get_environment(env_id)

        # 3. 检查是否需要开始新 episode（当前观察为 None 表示上个 episode 结束了）
        if env_data["current_obs"] is None:
            self._reset_environment(env_id)
            env_data = self._get_environment(env_id)  # 重新获取更新后的数据

        # 3. 获取当前环境观察
        wrapped_obs = env_data["env_wrapper"].wrap_observation(env_data["current_obs"])

        # 4. 生成 LLM 响应
        prompt_list = [{"role": "user", "content": wrapped_obs}]
        formatted_text = self.tokenizer.apply_chat_template(
            prompt_list,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # 训练效率考虑，禁用思考模式
        )

        prompt_ids = self.tokenizer.encode(formatted_text, add_special_tokens=True)

        # LLM 生成响应
        response_ids = await self.server_manager.generate(
            request_id=f"{request_id}_{env_data['episode_id']}_step{env_data['episode_step_count']}",
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )

        # 解码响应
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 5. 执行环境交互
        try:
            action_id = env_data["env_wrapper"].parse_llm_response(response_text)
        except Exception as e:
            print(f"⚠️ Action parsing failed: {e}, using default DO action")
            action_id = 5  # 默认 DO 动作

        # 执行动作并更新持久化的环境状态
        new_obs, new_state, reward, done, _ = env_data["env_wrapper"].step(
            env_data["current_state"], action_id
        )

        # 更新持久化的环境状态
        env_data["current_obs"] = new_obs
        env_data["current_state"] = new_state
        env_data["episode_step_count"] += 1

        # 更新累积奖励
        episode_id = env_data["episode_id"]
        if episode_id not in self.__class__._episode_cumulative_rewards:
            self.__class__._episode_cumulative_rewards[episode_id] = 0.0
        self.__class__._episode_cumulative_rewards[episode_id] += reward

        # 检查是否需要重置环境 (环境返回done=True 或 超过最大步数限制)
        max_steps_reached = (
            env_data["episode_step_count"] >= self.__class__._max_episode_steps
        )
        # check global step
        if done or max_steps_reached:
            cumulative_reward = self.__class__._episode_cumulative_rewards[episode_id]
            end_reason = (
                "environment done"
                if done
                else f"max steps ({self.__class__._max_episode_steps}) reached"
            )
            print(
                f"🏁 Episode {episode_id} finished after {env_data['episode_step_count']} steps ({end_reason}), cumulative reward: {cumulative_reward:.3f}"
            )

            # 计算reward占最大reward的比例
            reward_percentage = (
                cumulative_reward / self.__class__._max_craftax_reward * 100.0
            )

            # 直接记录到wandb
            try:
                import wandb

                if wandb.run is not None:
                    # 尝试从Verl训练框架获取当前global_step
                    current_global_step = self._get_current_global_step()

                    # 计算全局环境交互步数: (update_step * rollout_n + rollout_step) * num_envs
                    # 简化版本：假设rollout_step=0（批次开始）
                    global_env_steps = (
                        current_global_step
                        * self.__class__._rollout_n
                        * self.__class__._num_envs
                    )

                    wandb.log(
                        {
                            "craftax/reward_percentage": reward_percentage,
                            "craftax/cumulative_reward": cumulative_reward,
                        },
                        step=global_env_steps,
                    )
                    print(
                        f"📊 Wandb logged: Global step {current_global_step}, Env steps {global_env_steps}"
                    )
                    print(
                        f"    Reward: {cumulative_reward:.3f}, Percentage: {reward_percentage:.2f}%"
                    )
                else:
                    print(
                        f"📊 No wandb run active, Episode {episode_id}, Reward: {cumulative_reward:.3f}, Percentage: {reward_percentage:.2f}%"
                    )
            except ImportError:
                print(
                    f"📊 Wandb not available, Episode {episode_id}, Reward: {cumulative_reward:.3f}, Percentage: {reward_percentage:.2f}%"
                )

            # 清理已完成episode的奖励记录
            del self.__class__._episode_cumulative_rewards[episode_id]

            # 环境结束，标记需要重置（但不销毁环境）
            env_data["current_obs"] = None
            env_data["current_state"] = None

        # 5. 在 response 后面加上当前步骤的奖励信息
        reward_info = f" [Reward: {reward:.3f}]"
        reward_tokens = self.tokenizer.encode(reward_info, add_special_tokens=False)
        response_ids.extend(reward_tokens)

        # 创建响应掩码
        response_mask = [1] * len(response_ids)

        # 6. 构建最终输出
        # 使用配置中的序列长度限制
        final_prompt_ids = prompt_ids[: self.prompt_length]
        final_response_ids = response_ids[: self.response_length]
        final_response_mask = response_mask[: len(final_response_ids)]

        # 构建指标
        total_time = time.time() - start_time
        metrics = {
            "generate_sequences": total_time * 0.8,  # 大部分时间在生成
            "tool_calls": total_time * 0.2,  # 少部分时间在环境交互
            "step_reward": reward,  # 当前步奖励
            "episode_step": env_data["episode_step_count"],  # 当前 episode 步数
            "done": done,  # 是否结束
            "action_id": action_id,  # 执行的动作
            "episode_id": str(env_data["episode_id"]),  # episode 标识
            "env_id": env_id,  # 环境标识
        }

        # 如果episode结束，添加wandb记录数据
        if done:
            cumulative_reward = self.__class__._episode_cumulative_rewards.get(
                episode_id, 0.0
            )
            reward_percentage = (
                cumulative_reward / self.__class__._max_craftax_reward * 100.0
            )
            metrics.update(
                {
                    "episode_finished": True,
                    "cumulative_reward": cumulative_reward,
                    "reward_percentage": reward_percentage,  # 这是我们要绘制的纵坐标
                }
            )

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids,
            response_mask=final_response_mask,
            num_turns=1,  # 每次只执行一步
            metrics=AgentLoopMetrics(**metrics),
        )
