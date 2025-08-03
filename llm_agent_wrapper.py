#!/usr/bin/env python3
"""
Craftax LLM Agent 包装器
为LLM提供友好的观察格式和动作接口
"""

import os

# 让JAX使用GPU 6,7，Verl训练使用GPU 0-5

import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_text
from craftax.craftax.constants import Action
from typing import Dict, List, Tuple, Optional


class CraftaxLLMWrapper:
    """Craftax环境的LLM友好包装器"""

    def __init__(self, env_name="Craftax-Symbolic-v1", max_episode_steps=400):
        self.env = make_craftax_env_from_name(env_name, auto_reset=True)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # 动作映射
        self.action_names = [action.name.lower().replace("_", " ") for action in Action]

    def _get_action_list(self) -> List[str]:
        """获取简洁的动作列表"""
        return [action.name.lower().replace("_", " ") for action in Action]

    def _format_available_actions(self) -> str:
        """格式化可用动作列表"""
        action_list = []
        for i, name in enumerate(self.action_names):
            action_list.append(f"{i:2d}. {name}")

        return "\n".join(action_list)

    def _extract_inventory_items(self, inventory_text: str) -> str:
        """提取物品清单"""
        items = [
            "Wood:",
            "Stone:",
            "Coal:",
            "Iron:",
            "Diamond:",
            "Sapphire:",
            "Ruby:",
            "Sapling:",
            "Torch:",
            "Arrow:",
            "Book:",
            "potion",
        ]

        lines = inventory_text.split("\n")
        item_lines = []

        for line in lines:
            for item in items:
                if item in line:
                    item_lines.append(line.strip())
                    break

        return "\n".join(item_lines)

    def _extract_character_status(self, inventory_text: str) -> str:
        """提取角色状态"""
        status_items = ["Health:", "Food:", "Drink:", "Energy:"]

        lines = inventory_text.split("\n")
        status_lines = []

        for line in lines:
            for item in status_items:
                if item in line:
                    status_lines.append(line.strip())
                    break

        return "\n".join(status_lines)

    def _extract_character_attributes(self, inventory_text: str) -> str:
        """提取角色属性与成长"""
        attr_items = ["Mana:", "XP:", "Dexterity:", "Strength:", "Intelligence:"]

        lines = inventory_text.split("\n")
        attr_lines = []

        for line in lines:
            for item in attr_items:
                if item in line:
                    attr_lines.append(line.strip())
                    break

        return "\n".join(attr_lines)

    def _extract_environment_status(self, inventory_text: str) -> str:
        """提取当前环境与状态"""
        env_items = ["Direction:", "Light:", "Is Sleeping:", "Is Resting:"]

        lines = inventory_text.split("\n")
        env_lines = []

        for line in lines:
            for item in env_items:
                if item in line:
                    env_lines.append(line.strip())
                    break

        return "\n".join(env_lines)

    def _extract_world_status(self, inventory_text: str) -> str:
        """提取游戏进程与世界状态"""
        world_items = [
            "Learned Fireball:",
            "Learned Iceball:",
            "Floor:",
            "Ladder Open:",
            "Is Boss Vulnerable:",
        ]

        lines = inventory_text.split("\n")
        world_lines = []

        for line in lines:
            for item in world_items:
                if item in line:
                    world_lines.append(line.strip())
                    break

        return "\n".join(world_lines)

    def wrap_observation(self, text_obs: str) -> str:
        """包装观察为LLM友好的格式"""
        # 构建LLM提示
        prompt = "You are playing Craftax, a survival and crafting game.\n\n"

        # 游戏进度
        prompt += f"=== GAME STATUS (Step {self.current_step}) ===\n"

        # 完整地图
        map_section = (
            text_obs.split("Inventory:")[0] if "Inventory:" in text_obs else text_obs
        )
        prompt += f"=== MAP ===\n{map_section}\n"

        # 分类显示库存信息
        if "Inventory:" in text_obs:
            inventory_section = text_obs.split("Inventory:")[1]

            # 物品清单
            items = self._extract_inventory_items(inventory_section)
            if items:
                prompt += f"=== INVENTORY ITEMS ===\n{items}\n\n"

            # 角色状态
            character_status = self._extract_character_status(inventory_section)
            if character_status:
                prompt += f"=== CHARACTER STATUS ===\n{character_status}\n\n"

            # 角色属性与成长
            attributes = self._extract_character_attributes(inventory_section)
            if attributes:
                prompt += f"=== CHARACTER ATTRIBUTES ===\n{attributes}\n\n"

            # 当前环境与状态
            environment = self._extract_environment_status(inventory_section)
            if environment:
                prompt += f"=== ENVIRONMENT STATUS ===\n{environment}\n\n"

            # 游戏进程与世界状态
            world_status = self._extract_world_status(inventory_section)
            if world_status:
                prompt += f"=== WORLD STATUS ===\n{world_status}\n\n"

        # 可用动作
        prompt += "=== AVAILABLE ACTIONS ===\n"
        prompt += self._format_available_actions()
        prompt += "\n\n"

        # 最终指令
        prompt += "After a short reasoning process,show your next action in format of '[The reasoning process] Action: {action_name} (ID: {action_id})' .\n"
        return prompt

    def parse_llm_response(self, response: str) -> int:
        """解析LLM响应，返回动作ID，优先识别'Action:'后面的内容，如果有ID直接返回ID"""
        import re

        response = response.strip().lower()

        # 优先查找 'action:' 后面的内容
        action_pattern = r"action\s*:\s*([^\n\(]+)"
        id_pattern = r"\(id:\s*(\d+)\)"
        # 先查找 (ID: xxx)
        id_match = re.search(id_pattern, response)
        if id_match:
            action_id = int(id_match.group(1))
            if 0 <= action_id < len(self.action_names):
                return action_id

        # 再查找 'action:' 后面的内容
        match = re.search(action_pattern, response)
        action_str = None
        if match:
            action_str = match.group(1).strip()
            # 如果有 (ID: xxx) 这样的内容，去掉
            action_str = re.sub(r"\(id:.*?\)", "", action_str).strip()

        # 如果找到了 action_str，尝试用它做mapping
        if action_str:
            # 先尝试精确匹配动作名称
            for i, action_name in enumerate(self.action_names):
                if action_str == action_name.lower():
                    return i
            # 再尝试包含匹配
            for i, action_name in enumerate(self.action_names):
                if action_str in action_name.lower():
                    return i
            # 再尝试数字
            numbers = re.findall(r"\d+", action_str)
            if numbers:
                action_id = int(numbers[0])
                if 0 <= action_id < len(self.action_names):
                    return action_id

        # 如果没有找到 'Action:'，回退到原有逻辑
        # 尝试解析数字
        try:
            numbers = re.findall(r"\d+", response)
            if numbers:
                action_id = int(numbers[0])
                if 0 <= action_id < len(self.action_names):
                    return action_id
        except:
            pass

        # 尝试匹配动作名称
        for i, action_name in enumerate(self.action_names):
            if action_name in response:
                return i

        # 关键词映射
        keyword_mapping = {
            "noop": 0,
            "left": 1,
            "right": 2,
            "up": 3,
            "down": 4,
            "do": 5,
            "sleep": 6,
            "rest": 17,
            "descend": 18,
            "ascend": 19,
            "place stone": 7,
            "place table": 8,
            "place furnace": 9,
            "place plant": 10,
            "place torch": 28,
            "make wood pickaxe": 11,
            "make stone pickaxe": 12,
            "make iron pickaxe": 13,
            "make diamond pickaxe": 20,
            "make wood sword": 14,
            "make stone sword": 15,
            "make iron sword": 16,
            "make diamond sword": 21,
            "make arrow": 25,
            "make torch": 38,
            "make iron armour": 22,
            "make diamond armour": 23,
            "shoot arrow": 24,
            "cast fireball": 26,
            "cast iceball": 27,
            "drink potion red": 29,
            "drink potion green": 30,
            "drink potion blue": 31,
            "drink potion pink": 32,
            "drink potion cyan": 33,
            "drink potion yellow": 34,
            "read book": 35,
            "enchant sword": 36,
            "enchant armour": 37,
            "enchant bow": 42,
            "level up dexterity": 39,
            "level up strength": 40,
            "level up intelligence": 41,
            "move": 5,
            "walk": 5,
            "interact": 5,
            "use": 5,
            "attack": 5,
            "craft": 11,
            "make": 11,
            "build": 8,
            "place": 7,
            "put": 7,
            "drink": 29,
            "potion": 29,
            "cast": 26,
            "magic": 26,
            "spell": 26,
            "enchant": 36,
            "upgrade": 39,
            "levelup": 39,
        }

        raise ValueError("无法从响应中解析出有效的动作ID或动作名称。请检查输入内容。")

    def reset(self, seed: Optional[int] = None):
        """重置环境"""
        if seed is not None:
            rng = jax.random.PRNGKey(seed)
        else:
            rng = jax.random.PRNGKey(42)

        obs, state = self.env.reset(rng, self.env.default_params)
        self.current_step = 0

        # 获取text观察并包装
        text_obs = render_craftax_text(state)
        wrapped_obs = self.wrap_observation(text_obs)

        return wrapped_obs, state

    def step(self, state, action_id: int):
        """执行动作"""
        rng = jax.random.PRNGKey(self.current_step + 100)

        # 执行动作
        obs, new_state, reward, done, info = self.env.step(
            rng, state, action_id, self.env.default_params
        )

        self.current_step += 1

        # 检查是否达到最大步数
        if self.current_step >= self.max_episode_steps:
            done = True

        # 获取text观察并包装
        text_obs = render_craftax_text(new_state)
        wrapped_obs = self.wrap_observation(text_obs)

        return wrapped_obs, new_state, reward, done, info


def demo_wrapper():
    """演示包装器的使用"""
    print("=" * 80)
    print("Craftax LLM Agent 包装器演示")
    print("=" * 80)

    # 创建包装器
    wrapper = CraftaxLLMWrapper()

    # 重置环境
    wrapped_obs, state = wrapper.reset(seed=42)

    print("🎮 包装后的LLM观察:\n")
    print(wrapped_obs)

    # print("\n" + "=" * 60)
    # print("🤖 模拟LLM响应解析")
    # print("=" * 60)

    # # 测试不同的LLM响应格式
    # test_responses = [
    #     "I choose action 5 (do) to interact with nearby objects",
    #     "Let me move right to explore. Action: right",
    #     "I want to chop trees, so I'll use the do action",
    #     "Action 1 - move left",
    #     "5",
    #     "do"
    # ]

    # for response in test_responses:
    #     action_id = wrapper.parse_llm_response(response)
    #     action_name = wrapper.action_names[action_id]
    #     print(f"LLM响应: '{response}'")
    #     print(f"解析结果: {action_id} ({action_name})")
    #     print()


if __name__ == "__main__":
    demo_wrapper()
