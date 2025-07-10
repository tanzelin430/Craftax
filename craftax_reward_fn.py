#!/usr/bin/env python3
"""
Craftax Reward Function for Verl
从 LLM 的响应中提取环境奖励
"""

import re
from typing import Dict, Any, Union


def craftax_reward_function(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict[str, Any] = None, **kwargs) -> Union[float, Dict[str, Any]]:
    """
    从 Craftax Agent Loop 的响应中提取单步奖励
    
    Args:
        data_source: 数据源标识符（从数据集中获取）
        solution_str: LLM 生成的响应字符串
        ground_truth: 占位符，对于 Craftax 不使用
        extra_info: 额外信息
        **kwargs: 其他参数
        
    Returns:
        float: 提取的单步奖励值
    """
    
    # 从响应字符串中提取单步奖励信息
    # 查找 "[Reward: Y.YYY]" 格式的文本（新格式）
    reward_match = re.search(r'\[Reward:\s*([-+]?\d*\.?\d+)\]', solution_str)
    
    if reward_match:
        step_reward = float(reward_match.group(1))
        
        # 记录调试信息
        if extra_info is not None:
            extra_info.update({
                "step_reward": step_reward,
                "response_length": len(solution_str)
            })
        
        print(f"[Craftax Reward] Single step reward: {step_reward:.3f}")
        # show last 20 token of solution_str
        # print(f"[Craftax Reward] Last 50 tokens of solution_str: {solution_str}")
        return step_reward
    else:
        # # 兼容旧格式 "[Step X Reward: Y.YY]"（如果存在）
        # old_format_match = re.search(r'\[Step\s+\d+\s+Reward:\s*([-+]?\d*\.?\d+)\]', solution_str)
        # if old_format_match:
        #     step_reward = float(old_format_match.group(1))
        #     print(f"[Craftax Reward] Found old format reward: {step_reward:.3f}")
        #     return step_reward
        
        raise ValueError(f"[Craftax Reward] 未在响应中找到奖励信息: {solution_str}...")
        return 0.0