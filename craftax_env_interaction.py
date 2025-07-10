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
from typing import Dict, List, Any

from llm_agent_wrapper import CraftaxLLMWrapper


class CraftaxQwenAgent:
    """
    使用 Qwen3-14B 模型的 Craftax 智能体
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化智能体
        
        Args:
            model_path: Qwen3-14B 模型路径
            device: 设备类型
        """
        self.device = device
        self.model_path = model_path
        
        # 初始化 Craftax 环境包装器
        self.env_wrapper = CraftaxLLMWrapper(
            env_name='Craftax-Symbolic-v1',
            max_episode_steps=500
        )
        
        # 初始化模型和tokenizer
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 添加pad_token如果不存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.eval()
        
        print("Model and environment initialized successfully!")
    
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
            enable_thinking=False # Switch between thinking and non-thinking modes. Default is True. We need it to be False for training efficiency.
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
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码响应（只取新生成的部分）
        response = self.tokenizer.decode(
            outputs[0][model_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
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
            wrapped_obs, state, reward, done, info = self.env_wrapper.step(state, action_id)
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
                "llm_response": llm_response
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
            "average_reward": float(total_reward / step_count) if step_count > 0 else 0.0,
            "history": episode_history
        }
        
        if verbose:
            print(f"\n📈 Episode Summary:")
            print(f"   Total Steps: {step_count}")
            print(f"   Total Reward: {total_reward:.2f}")
            print(f"   Average Reward: {episode_stats['average_reward']:.2f}")
            print(f"   Completed: {done}")
        
        return episode_stats
    
    def run_multiple_episodes(self, num_episodes: int = 5, max_steps_per_episode: int = 100) -> Dict[str, Any]:
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
            print('='*60)
            
            episode_stats = self.run_episode(
                max_steps=max_steps_per_episode,
                verbose=True
            )
            
            all_episodes.append(episode_stats)
            total_rewards.append(episode_stats['total_reward'])
        
        # 计算总体统计
        overall_stats = {
            "num_episodes": num_episodes,
            "episodes": all_episodes,
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "max_reward": float(np.max(total_rewards)),
            "min_reward": float(np.min(total_rewards)),
            "completion_rate": float(sum(1 for ep in all_episodes if ep['completed']) / num_episodes)
        }
        
        print(f"\n{'='*60}")
        print("📊 OVERALL STATISTICS")
        print('='*60)
        print(f"Episodes: {num_episodes}")
        print(f"Mean Reward: {overall_stats['mean_reward']:.2f} ± {overall_stats['std_reward']:.2f}")
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
        # 创建智能体
        agent = CraftaxQwenAgent(model_path=model_path)
        
        # 运行多个episodes
        results = agent.run_multiple_episodes(
            num_episodes=1,
            max_steps_per_episode=5
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
            elif hasattr(obj, '__array__'):  # JAX数组或numpy数组
                try:
                    return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
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