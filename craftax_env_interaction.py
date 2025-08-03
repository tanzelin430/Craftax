#!/usr/bin/env python3
"""
Craftax Environment Interaction Script
使用 Qwen3-14B 模型与 Craftax 环境进行交互
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from typing import Dict, List, Any

from llm_agent_wrapper import CraftaxLLMWrapper


class CraftaxQwenAgent:
    """
    使用 Qwen3-14B 模型的 Craftax 智能体
    """

    def __init__(
        self, model_path: str, device: str = "cuda", num_parallel_envs: int = 1
    ):
        """
        初始化智能体 - 模拟craftax_agent_loop.py的环境管理方式

        Args:
            model_path: Qwen3-14B 模型路径
            device: 设备类型
            num_parallel_envs: 并行环境数量（模拟craftax_agent_loop.py）
        """
        init_start_time = time.time()

        self.device = device
        self.model_path = model_path
        self.num_parallel_envs = num_parallel_envs

        # 1. 环境创建时间
        print(f"🚀 [ENV CREATE START] Creating {num_parallel_envs} environments...")
        env_create_start = time.time()

        self.env_wrapper = CraftaxLLMWrapper(
            env_name="Craftax-Symbolic-v1", max_episode_steps=500
        )

        self._environments = {}  # 模拟类级别环境存储
        for i in range(num_parallel_envs):
            env_id = f"env_{i}"
            self._create_environment(env_id)

        env_create_time = time.time() - env_create_start
        print(
            f"✅ [ENV CREATE END] {num_parallel_envs} environments created in {env_create_time:.3f}s"
        )

        # 初始化模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )

        # 添加pad_token如果不存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.model.eval()
        print(f"Model loaded successfully!")

    def _create_environment(self, env_id: str):
        """模拟craftax_agent_loop.py: 创建并初始化一个新环境"""
        # 创建环境状态存储（简化版）
        seed = np.random.randint(0, 100000)
        wrapped_obs, state = self.env_wrapper.reset(seed=seed)

        self._environments[env_id] = {
            "current_state": state,
            "current_obs": wrapped_obs,
            "episode_step_count": 0,
            "episode_id": f"ep_{env_id}_{seed}",
            "seed": seed,
        }

    def _get_environment(self, env_id: str):
        """模拟craftax_agent_loop.py: 获取指定的预创建环境"""
        if env_id not in self._environments:
            self._create_environment(env_id)
        return self._environments[env_id]

    def generate_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        使用模型生成响应
        """
        # 2. 每一轮rollout推理时间
        rollout_start = time.time()

        prompt_list = [{"role": "assistant", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            prompt_list,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][model_inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        rollout_time = time.time() - rollout_start
        print(f"🧠 [ROLLOUT] Inference time: {rollout_time:.3f}s")

        return response.strip()

    def run_episode(self, max_steps: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """
        运行一个完整的episode

        Args:
            max_steps: 最大步数
            verbose: 是否打印详细信息

        Returns:
            episode结果统计
        """
        # 重置环境
        wrapped_obs, state = self.env_wrapper.reset(seed=np.random.randint(0, 10000))

        total_reward = 0.0
        step_count = 0
        done = False

        episode_history = []

        if verbose:
            print("=" * 80)
            print("🎮 Starting New Craftax Episode")
            print("=" * 80)

        while not done and step_count < max_steps:
            if verbose:
                print(f"\n📍 Step {step_count + 1}")
                print("-" * 40)

            # 获取LLM响应
            llm_response = self.generate_response(wrapped_obs)

            if verbose:
                print(f"🤖 LLM Response: {llm_response}")

            # 3. 每一轮batch step的交互时间
            step_start = time.time()

            # 解析动作
            action_id = self.env_wrapper.parse_llm_response(llm_response)
            action_name = self.env_wrapper.action_names[action_id]

            # 执行动作
            wrapped_obs, state, reward, done, info = self.env_wrapper.step(
                state, action_id
            )
            wrapped_obs = self.env_wrapper.wrap_observation(wrapped_obs)

            step_time = time.time() - step_start
            print(f"🔄 [STEP] Environment interaction time: {step_time:.3f}s")
            # print action name and id
            print(f"⚡ Action: {action_name} (ID: {action_id})")

            total_reward += reward
            step_count += 1

            if verbose:
                print(f"⚡ Action: {action_name} (ID: {action_id})")

            # 记录步骤信息
            step_info = {
                "step": step_count,
                "action_id": action_id,
                "action_name": action_name,
                "reward": float(reward),
                "done": done,
                "llm_response": llm_response,
            }
            episode_history.append(step_info)

            if verbose:
                print(f"💰 Reward: {reward:.2f}")
                print(f"📊 Total Reward: {total_reward:.2f}")
                if done:
                    print("✅ Episode Complete!")

        # 返回episode统计
        episode_stats = {
            "total_reward": float(total_reward),
            "total_steps": step_count,
            "completed": done,
            "average_reward": float(total_reward / step_count)
            if step_count > 0
            else 0.0,
            "history": episode_history,
        }

        if verbose:
            print(f"\n📈 Episode Summary:")
            print(f"   Total Steps: {step_count}")
            print(f"   Total Reward: {total_reward:.2f}")
            print(f"   Average Reward: {episode_stats['average_reward']:.2f}")
            print(f"   Completed: {done}")

        return episode_stats

    def run_serial_batch_like_verl_worker(
        self, batch_size: int = 8, steps_per_batch: int = 5
    ) -> Dict[str, Any]:
        """
        模拟Verl单个worker内的串行批处理：
        - 管理batch_size个环境
        - 每个环境串行执行steps_per_batch步
        - 模拟真实的worker内执行模式
        """
        print(
            f"🏭 [WORKER SIMULATION] Processing {batch_size} environments serially (like single Verl worker)"
        )

        total_rollout_time = 0
        total_step_time = 0
        batch_results = []

        # 串行处理每个环境
        for env_idx in range(batch_size):
            env_id = f"env_{env_idx}"
            env_data = self._get_environment(env_id)

            print(
                f"\n🎮 Processing Environment {env_idx + 1}/{batch_size} (env_id: {env_id})"
            )

            # 如果环境需要重置
            if env_data["current_obs"] is None:
                seed = np.random.randint(0, 100000)
                wrapped_obs, state = self.env_wrapper.reset(seed=seed)
                env_data["current_state"] = state
                env_data["current_obs"] = wrapped_obs
                env_data["episode_step_count"] = 0
                print(f"🔄 Reset environment {env_id} with seed {seed}")

            episode_reward = 0
            step_count = 0

            # 串行执行多步
            for step in range(steps_per_batch):
                print(f"📍 Step {step + 1}/{steps_per_batch}")

                # 模拟agent_loop的单步处理
                wrapped_obs = env_data["current_obs"]

                # LLM推理
                llm_response = self.generate_response(wrapped_obs)
                total_rollout_time += (
                    time.time() - time.time()
                )  # 这里rollout时间已在generate_response中统计

                # 环境交互
                step_start = time.time()
                try:
                    action_id = self.env_wrapper.parse_llm_response(llm_response)
                except:
                    action_id = 0

                new_obs, new_state, reward, done, _ = self.env_wrapper.step(
                    env_data["current_state"], action_id
                )
                wrapped_obs = self.env_wrapper.wrap_observation(new_obs)

                step_time = time.time() - step_start
                total_step_time += step_time
                print(f"🔄 [STEP] Environment interaction time: {step_time:.3f}s")
                print(f"⚡ LLM Response: {llm_response}")
                # 更新环境状态
                env_data["current_state"] = new_state
                env_data["current_obs"] = wrapped_obs
                env_data["episode_step_count"] += 1
                episode_reward += reward
                step_count += 1

                print(f"💰 Reward: {reward:.2f}, Action: {action_id}")

                if done:
                    print(f"✅ Episode finished!")
                    env_data["current_obs"] = None  # 标记需要重置
                    break

            batch_results.append(
                {
                    "env_id": env_id,
                    "episode_reward": episode_reward,
                    "steps_completed": step_count,
                }
            )

        # 统计结果
        total_reward = sum(r["episode_reward"] for r in batch_results)
        total_steps = sum(r["steps_completed"] for r in batch_results)

        print(f"\n📊 [BATCH SUMMARY]")
        print(f"Total environments processed: {batch_size}")
        print(f"Total steps executed: {total_steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward per env: {total_reward/batch_size:.2f}")
        print(f"Average steps per env: {total_steps/batch_size:.1f}")

        return {
            "batch_size": batch_size,
            "total_steps": total_steps,
            "total_reward": total_reward,
            "results": batch_results,
            "total_step_time": total_step_time,
        }

    def run_multiple_episodes(
        self, num_episodes: int = 5, max_steps_per_episode: int = 100
    ) -> Dict[str, Any]:
        """向后兼容的接口"""
        return self.run_serial_batch_like_verl_worker(
            batch_size=num_episodes, steps_per_batch=max_steps_per_episode
        )


def main():
    """主函数"""
    # 模型路径
    model_path = "/fs-computility/mabasic/shared/models/Qwen3-8B"

    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print("Please check the model path and try again.")
        return

    try:
        # 创建智能体
        agent = CraftaxQwenAgent(model_path=model_path, num_parallel_envs=50)  # 预创建8个环境

        # 运行测试
        results = agent.run_multiple_episodes(
            num_episodes=50, max_steps_per_episode=500
        )

        # 保存结果
        import json

        # 转换JAX数组为Python原生类型以便JSON序列化
        def convert_jax_arrays(obj):
            """递归转换JAX数组为Python原生类型"""
            if isinstance(obj, dict):
                return {key: convert_jax_arrays(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_jax_arrays(item) for item in obj]
            elif hasattr(obj, "__array__"):  # JAX数组或numpy数组
                try:
                    return obj.tolist() if hasattr(obj, "tolist") else float(obj)
                except:
                    return str(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # 转换结果
        serializable_results = convert_jax_arrays(results)

        with open("craftax_experiment_results_4B_LT.json", "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n✅ Results saved to craftax_experiment_results.json")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
