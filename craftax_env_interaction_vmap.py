#!/usr/bin/env python3
"""
Craftax Environment Interaction Script
使用 Qwen3-14B 模型与 Craftax 环境进行交互
"""

import os

# 禁用flash attention
os.environ["FLASH_ATTENTION_DISABLE"] = "1"

# 配置JAX显存管理
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # 禁用预分配
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"  # 限制显存使用为30%
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # 使用平台分配器

import jax
import jax.numpy as jnp

# 配置JAX显存增量分配
jax.config.update("jax_enable_x64", False)  # 使用32位减少显存
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List, Any, Optional
from functools import partial

from llm_agent_wrapper import CraftaxLLMWrapper


class CraftaxQwenAgent:
    """
    使用 Qwen3-14B 模型的 Craftax 智能体
    支持GPU加速和真正的JAX并行化
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        num_parallel_envs: int = 1,
        env_device: str = "auto",
    ):
        """
        初始化智能体

        Args:
            model_path: Qwen3-14B 模型路径
            device: 设备类型
        """
        self.device = device
        self.model_path = model_path
        self.num_parallel_envs = num_parallel_envs

        # 检查JAX设备和GPU状态
        self._check_jax_gpu_setup()

        # 初始化 Craftax 环境包装器
        self.env_wrapper = CraftaxLLMWrapper(
            env_name="Craftax-Symbolic-v1", max_episode_steps=500
        )

        # 创建并行环境的JAX函数
        if self.has_gpu:
            self._setup_parallel_env_functions()

        # 初始化模型和tokenizer
        print(f"Loading model from {model_path}")
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
            device_map="auto",  # 自动分配GPU
            attn_implementation="eager",  # 使用eager attention避免flash-attn问题
        )

        self.model.eval()

        print(f"Model and environment initialized successfully!")
        print(f"JAX devices: {jax.devices()}")
        print(f"Number of parallel environments: {self.num_parallel_envs}")

    def generate_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        使用模型生成响应

        Args:
            prompt: 输入提示
            max_new_tokens: 最大新token数

        Returns:
            生成的响应文本
        """
        prompt_list = [{"role": "assistant", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            prompt_list,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Switch between thinking and non-thinking modes. Default is True. We need it to be False for training efficiency.
        )
        # print(f"text: {text}")
        # 2. 将格式化后的 text 进行 tokenize
        #    注意：这里我们直接使用格式化好的 text
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # 3. 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                # 4. 确保传递的是格式化后的 model_inputs
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 解码响应（只取新生成的部分）
        response = self.tokenizer.decode(
            outputs[0][model_inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response.strip()

    def _check_jax_gpu_setup(self):
        """检查JAX GPU设置"""
        devices = jax.devices()
        print(f"Available JAX devices: {devices}")

        # 检查是否有GPU - JAX CUDA设备的平台名称是'gpu'
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if gpu_devices:
            print(f"Found {len(gpu_devices)} GPU devices: {gpu_devices}")
            self.has_gpu = True
        else:
            print("No GPU devices found, using CPU")
            self.has_gpu = False

    def _setup_parallel_env_functions(self):
        """设置并行环境的JAX函数"""
        # 获取原始环境
        from craftax.craftax_env import make_craftax_env_from_name

        self.base_env = make_craftax_env_from_name(
            "Craftax-Symbolic-v1", auto_reset=True
        )

        # 使用JAX的vmap创建并行环境重置函数
        @partial(jax.jit)
        def parallel_reset(rng_keys):
            """并行重置多个环境"""

            def single_reset(rng):
                return self.base_env.reset(rng, self.base_env.default_params)

            return jax.vmap(single_reset)(rng_keys)

        # 使用JAX的vmap创建并行环境步进函数
        @partial(jax.jit)
        def parallel_step(rng_keys, states, actions):
            """并行执行多个环境的步进"""

            def single_step(rng, state, action):
                return self.base_env.step(
                    rng, state, action, self.base_env.default_params
                )

            return jax.vmap(single_step)(rng_keys, states, actions)

        self.parallel_reset_fn = parallel_reset
        self.parallel_step_fn = parallel_step

        print(f"✅ Parallel environment functions created with vmap")
        print(f"🚀 JAX will automatically use GPU if available: {self.has_gpu}")

        # 预编译JAX函数，避免运行时编译延迟
        self._warmup_jax_functions()

    def _warmup_jax_functions(self):
        """预编译JAX函数，避免运行时编译延迟导致GPU空闲"""
        print("🔥 Warming up JAX functions to reduce runtime compilation delays...")

        # 创建小批量数据用于预热
        warmup_batch_size = min(8, self.num_parallel_envs)

        # 生成预热用的随机键
        warmup_rng = jax.random.PRNGKey(12345)
        warmup_keys = jax.random.split(warmup_rng, warmup_batch_size)

        # 预热reset函数
        print("  - Warming up parallel_reset_fn...")
        try:
            warmup_obs, warmup_states = self.parallel_reset_fn(warmup_keys)
            print(f"    ✅ Reset function warmed up with batch size {warmup_batch_size}")
        except Exception as e:
            print(f"    ⚠️ Reset warmup failed: {e}")

        # 预热step函数
        print("  - Warming up parallel_step_fn...")
        try:
            # 创建假的动作数组（全部是noop动作）
            warmup_actions = jnp.zeros(warmup_batch_size, dtype=jnp.int32)
            warmup_step_keys = jax.random.split(warmup_rng, warmup_batch_size)

            # 执行预热step
            _, _, _, _, _ = self.parallel_step_fn(
                warmup_step_keys, warmup_states, warmup_actions
            )
            print(f"    ✅ Step function warmed up with batch size {warmup_batch_size}")
        except Exception as e:
            print(f"    ⚠️ Step warmup failed: {e}")

        # 额外的预热：运行几个完整的step循环
        print("  - Running additional warmup cycles...")
        try:
            for warmup_cycle in range(3):
                cycle_keys = jax.random.split(warmup_rng, warmup_batch_size)
                warmup_actions = jnp.zeros(warmup_batch_size, dtype=jnp.int32)

                # 执行一个完整的循环
                _, new_states, _, _, _ = self.parallel_step_fn(
                    cycle_keys, warmup_states, warmup_actions
                )
                warmup_states = new_states

            print(f"    ✅ Completed {3} warmup cycles")
        except Exception as e:
            print(f"    ⚠️ Warmup cycles failed: {e}")

        print(
            "🔥 JAX function warmup completed - should reduce GPU idle time during execution!"
        )

    def generate_batch_responses(
        self, prompts: List[str], max_new_tokens: int = 200
    ) -> List[str]:
        """
        批量生成响应（使用GPU加速）

        Args:
            prompts: 输入提示列表
            max_new_tokens: 最大新token数

        Returns:
            生成的响应文本列表
        """
        if not prompts:
            return []

        # 准备批量输入
        batch_prompts = []
        for prompt in prompts:
            prompt_list = [{"role": "assistant", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                prompt_list,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            batch_prompts.append(text)

        # 批量tokenize
        model_inputs = self.tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # 批量生成（GPU加速）- 优化生成参数
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=1,  # 禁用beam search提高速度
                use_cache=True,  # 启用KV缓存
            )

        # 解码响应
        responses = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(
                output[model_inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            responses.append(response.strip())

        return responses

    def _create_multi_gpu_env_functions(self):
        """为每个GPU创建环境函数"""
        from craftax.craftax_env import make_craftax_env_from_name

        self.gpu_env_functions = {}

        for gpu_id, info in self.gpu_env_distribution.items():
            device = info["device"]
            num_envs = info["num_envs"]

            # 在特定GPU上创建环境函数
            with jax.default_device(device):
                base_env = make_craftax_env_from_name(
                    "Craftax-Symbolic-v1", auto_reset=True
                )

                # 创建该GPU的并行函数
                @partial(jax.jit, device=device)
                def gpu_parallel_reset(rng_keys):
                    def single_reset(rng):
                        return base_env.reset(rng, base_env.default_params)

                    return jax.vmap(single_reset)(rng_keys)

                @partial(jax.jit, device=device)
                def gpu_parallel_step(rng_keys, states, actions):
                    def single_step(rng, state, action):
                        return base_env.step(
                            rng, state, action, base_env.default_params
                        )

                    return jax.vmap(single_step)(rng_keys, states, actions)

                self.gpu_env_functions[gpu_id] = {
                    "reset_fn": gpu_parallel_reset,
                    "step_fn": gpu_parallel_step,
                    "num_envs": num_envs,
                }

        print(f"✅ Created environment functions for {len(self.gpu_env_functions)} GPUs")

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

            # 获取LLM响应 no limit on max new token
            llm_response = self.generate_response(wrapped_obs)

            if verbose:
                print(f"🤖 LLM Response: {llm_response}")

            # 解析动作
            action_id = self.env_wrapper.parse_llm_response(llm_response)
            action_name = self.env_wrapper.action_names[action_id]

            if verbose:
                print(f"⚡ Action: {action_name} (ID: {action_id})")

            # 执行动作
            wrapped_obs, state, reward, done, info = self.env_wrapper.step(
                state, action_id
            )
            wrapped_obs = self.env_wrapper.wrap_observation(wrapped_obs)
            total_reward += reward
            step_count += 1

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

    def run_parallel_episodes_gpu(
        self, num_episodes: int = 4, max_steps_per_episode: int = 100
    ) -> Dict[str, Any]:
        """
        使用JAX vmap在GPU上真正并行运行多个episodes

        Args:
            num_episodes: episode数量
            max_steps_per_episode: 每个episode的最大步数

        Returns:
            多episode统计结果
        """
        if num_episodes > self.num_parallel_envs:
            print(
                f"⚠️  Requested {num_episodes} episodes, but only {self.num_parallel_envs} parallel envs configured"
            )
            print(f"Running {self.num_parallel_envs} episodes in parallel")
            num_episodes = self.num_parallel_envs

        print(
            f"🚀 Running {num_episodes} episodes in TRUE parallel on GPU using JAX vmap"
        )

        # 创建不同的随机种子
        rng = jax.random.PRNGKey(42)
        rng_keys = jax.random.split(rng, num_episodes)

        # 并行重置所有环境 (GPU加速)
        print("🔄 Resetting environments in parallel on GPU...")
        obs_batch, states_batch = self.parallel_reset_fn(rng_keys)

        # 将JAX观察转换为文本观察
        from craftax.craftax.renderer import render_craftax_text

        text_observations = []
        for i in range(num_episodes):
            # 从批次中提取单个状态
            single_state = jax.tree.map(lambda x: x[i], states_batch)
            text_obs = render_craftax_text(single_state)
            wrapped_obs = self.env_wrapper.wrap_observation(text_obs)
            text_observations.append(wrapped_obs)

        # 初始化统计信息
        total_rewards = jnp.zeros(num_episodes)
        step_counts = jnp.zeros(num_episodes, dtype=jnp.int32)
        dones = jnp.zeros(num_episodes, dtype=jnp.bool_)
        episode_histories = [[] for _ in range(num_episodes)]

        # 并行执行步骤
        for step in range(max_steps_per_episode):
            if jnp.all(dones):
                break

            print(f"📍 Parallel Step {step + 1}/{max_steps_per_episode}")

            # 收集未完成环境的观察
            active_indices = jnp.where(~dones)[0]
            if len(active_indices) == 0:
                break

            active_observations = [text_observations[i] for i in active_indices]

            # 批量生成LLM响应 (GPU加速) - 总是使用批量处理
            print("⏱️  [GPU COMPUTE START] Batch LLM inference...")
            llm_responses = self.generate_batch_responses(active_observations)
            print("⏱️  [GPU COMPUTE END] Batch LLM inference completed")

            # 解析动作
            actions = []
            for i, response in enumerate(llm_responses):
                try:
                    action_id = self.env_wrapper.parse_llm_response(response)
                    actions.append(action_id)
                except:
                    actions.append(0)  # 默认动作

            # 为并行执行准备动作数组
            action_array = jnp.zeros(num_episodes, dtype=jnp.int32)
            for i, active_idx in enumerate(active_indices):
                action_array = action_array.at[active_idx].set(actions[i])

            # 创建新的随机键
            step_rng = jax.random.split(rng, num_episodes)

            # 并行执行环境步骤 (GPU加速)
            print("⏱️  [GPU COMPUTE START] Parallel environment step...")
            (
                new_obs_batch,
                new_states_batch,
                rewards_batch,
                new_dones_batch,
                info_batch,
            ) = self.parallel_step_fn(step_rng, states_batch, action_array)
            print("⏱️  [GPU COMPUTE END] Parallel environment step completed")

            # 更新统计信息
            # 只更新未完成的环境
            mask = ~dones
            total_rewards = jnp.where(
                mask, total_rewards + rewards_batch, total_rewards
            )
            step_counts = jnp.where(mask, step_counts + 1, step_counts)
            dones = jnp.where(mask, new_dones_batch, dones)

            # 更新状态 - 简化处理
            states_batch = new_states_batch

            # 更新文本观察
            for i in range(num_episodes):
                if not dones[i]:
                    single_state = jax.tree.map(lambda x: x[i], states_batch)
                    text_obs = render_craftax_text(single_state)
                    wrapped_obs = self.env_wrapper.wrap_observation(text_obs)
                    text_observations[i] = wrapped_obs

                    # 记录历史
                    if i < len(active_indices):
                        active_local_idx = jnp.where(active_indices == i)[0]
                        if len(active_local_idx) > 0:
                            local_idx = active_local_idx[0]
                            step_info = {
                                "step": int(step_counts[i]),
                                "action_id": int(actions[local_idx]),
                                "action_name": self.env_wrapper.action_names[
                                    actions[local_idx]
                                ],
                                "reward": float(rewards_batch[i]),
                                "done": bool(dones[i]),
                                "llm_response": llm_responses[local_idx],
                            }
                            episode_histories[i].append(step_info)

            # 打印进度
            active_count = jnp.sum(~dones)
            print(
                f"   Active environments: {active_count}, Avg reward: {jnp.mean(total_rewards):.2f}"
            )

        # 构建episode结果
        episodes = []
        for i in range(num_episodes):
            episode_stats = {
                "total_reward": float(total_rewards[i]),
                "total_steps": int(step_counts[i]),
                "completed": bool(dones[i]),
                "average_reward": float(total_rewards[i] / step_counts[i])
                if step_counts[i] > 0
                else 0.0,
                "history": episode_histories[i],
            }
            episodes.append(episode_stats)

        # 计算总体统计
        overall_stats = {
            "num_episodes": num_episodes,
            "episodes": episodes,
            "mean_reward": float(jnp.mean(total_rewards)),
            "std_reward": float(jnp.std(total_rewards)),
            "max_reward": float(jnp.max(total_rewards)),
            "min_reward": float(jnp.min(total_rewards)),
            "completion_rate": float(jnp.mean(dones)),
        }

        print(f"\n{'='*60}")
        print("🎯 GPU PARALLEL EXECUTION RESULTS")
        print("=" * 60)
        print(f"Episodes: {num_episodes}")
        print(
            f"Mean Reward: {overall_stats['mean_reward']:.2f} ± {overall_stats['std_reward']:.2f}"
        )
        print(f"Max Reward: {overall_stats['max_reward']:.2f}")
        print(f"Min Reward: {overall_stats['min_reward']:.2f}")
        print(f"Completion Rate: {overall_stats['completion_rate']:.1%}")

        return overall_stats

    def run_multiple_episodes(
        self, num_episodes: int = 5, max_steps_per_episode: int = 100
    ) -> Dict[str, Any]:
        """
        运行多个episodes

        Args:
            num_episodes: episode数量
            max_steps_per_episode: 每个episode的最大步数

        Returns:
            多episode统计结果
        """
        all_episodes = []
        total_rewards = []

        print(f"🚀 Running {num_episodes} episodes...")

        for episode_idx in range(num_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode_idx + 1}/{num_episodes}")
            print("=" * 60)

            episode_stats = self.run_episode(
                max_steps=max_steps_per_episode, verbose=True
            )

            all_episodes.append(episode_stats)
            total_rewards.append(episode_stats["total_reward"])

        # 计算总体统计
        overall_stats = {
            "num_episodes": num_episodes,
            "episodes": all_episodes,
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "max_reward": float(np.max(total_rewards)),
            "min_reward": float(np.min(total_rewards)),
            "completion_rate": float(
                sum(1 for ep in all_episodes if ep["completed"]) / num_episodes
            ),
        }

        print(f"\n{'='*60}")
        print("📊 OVERALL STATISTICS")
        print("=" * 60)
        print(f"Episodes: {num_episodes}")
        print(
            f"Mean Reward: {overall_stats['mean_reward']:.2f} ± {overall_stats['std_reward']:.2f}"
        )
        print(f"Max Reward: {overall_stats['max_reward']:.2f}")
        print(f"Min Reward: {overall_stats['min_reward']:.2f}")
        print(f"Completion Rate: {overall_stats['completion_rate']:.1%}")

        return overall_stats


def main():
    """主函数"""
    # 模型路径
    model_path = "/fs-computility/mabasic/shared/models/Qwen3-4B"

    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print("Please check the model path and try again.")
        return

    try:
        # 创建智能体（支持并行环境）
        agent = CraftaxQwenAgent(model_path=model_path, num_parallel_envs=20)  # 并行4个环境

        # 运行多GPU并行episodes
        results = agent.run_parallel_episodes_gpu(
            num_episodes=20, max_steps_per_episode=20
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
