#!/usr/bin/env python3
"""
Craftax PPO Training Script
使用 Verl 框架训练 Qwen3-14B 模型在 Craftax 环境中进行 PPO 训练
使用8张GPU进行分布式训练
"""

import os
import sys
import ray
import socket
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

# 添加当前目录到 Python 路径
sys.path.append("/root/work/Craftax")


def main():
    """主训练函数"""
    # 限制Verl只使用GPU 0-5，留GPU 6-7给JAX环境

    # 1. 加载配置
    config_path = "/root/work/Craftax/craftax_ppo_config.yaml"
    config = OmegaConf.load(config_path)
    print(f"✅ Configuration loaded: {config.trainer.project_name}")
    print(
        f"🎯 Using {config.trainer.n_gpus_per_node} GPUs on {config.trainer.nnodes} node(s)"
    )

    # 2. 初始化 Ray
    if not ray.is_initialized():
        # Check if we're connecting to an existing cluster
        ray_address = os.environ.get("RAY_ADDRESS")
        if ray_address:
            # When connecting to existing cluster, don't specify num_cpus/num_gpus
            ray.init(
                address=ray_address,
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN",
                        "VLLM_LOGGING_LEVEL": "INFO",
                        "VLLM_USE_V1": "1",
                    }
                },
            )
        else:
            # Starting new local cluster, can specify resources
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN",
                        "VLLM_LOGGING_LEVEL": "INFO",
                        "VLLM_USE_V1": "1",
                        "https_proxy": os.environ.get("https_proxy", ""),
                        "http_proxy": os.environ.get("http_proxy", ""),
                        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                    }
                },
                num_cpus=config.ray_init.num_cpus,
            )

    # 3. 创建并运行远程任务
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    print("✅ Training completed!")
    ray.shutdown()


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))

        # 解析配置
        OmegaConf.resolve(config)

        # 1. 设置模型路径
        model_path = config.actor_rollout_ref.model.path
        print(f"🤖 Loading model from: {model_path}")

        # 2. 加载 tokenizer 和 processor
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
        processor = hf_processor(model_path, trust_remote_code=True, use_fast=True)

        # 3. 定义 worker 类（使用 FSDP 策略）
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import (
            ActorRolloutRefWorker,
            AsyncActorRolloutRefWorker,
            CriticWorker,
        )

        actor_rollout_cls = (
            AsyncActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.mode == "async"
            else ActorRolloutRefWorker
        )
        ray_worker_group_cls = RayWorkerGroup

        # 4. 角色映射
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        import ray  # 确保ray在当前作用域中可用

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # 5. 资源池配置 - 8 GPUs 分布
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # 6. 创建数据集
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_craftax_dataset(
            config.data.train_files, config.data, tokenizer, processor
        )
        val_dataset = None

        # 7. 创建采样器
        train_sampler = create_sampler(config.data, train_dataset)

        # 8. 初始化 PPO 训练器
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        # 奖励函数通过配置文件的 custom_reward_function 字段加载
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=None,  # 奖励函数通过配置文件加载
            val_reward_fn=None,
            train_dataset=train_dataset,
            val_dataset=None,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )

        # 9. 初始化 workers 并开始训练
        print("🔧 Initializing workers...")
        trainer.init_workers()

        print("🎯 Starting PPO training...")
        trainer.fit()


def create_craftax_dataset(data_paths, data_config, tokenizer, processor):
    """创建 Craftax 数据集"""
    from torch.utils.data import Dataset
    from verl.utils.import_utils import load_extern_type

    # 加载自定义数据集类
    if (
        "custom_cls" in data_config
        and data_config.custom_cls.get("path", None) is not None
    ):
        dataset_cls = load_extern_type(
            data_config.custom_cls.path, data_config.custom_cls.name
        )
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"Custom dataset class must inherit from torch.utils.data.Dataset"
            )
    else:
        raise ValueError("Must specify custom_cls for Craftax dataset")

    print(f"📊 Using dataset class: {dataset_cls.__name__}")

    # 实例化数据集
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        num_episodes=data_config.get("num_episodes", 100),
    )

    print(f"📊 Dataset created with {len(dataset)} episodes")
    # print(f"show first element:{dataset.__getitem__(0)}")
    return dataset


def create_sampler(data_config, dataset):
    """创建数据采样器"""
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.get("shuffle", True):
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(
            data_source=dataset, generator=train_dataloader_generator
        )
        print("📊 Using RandomSampler")
    else:
        sampler = SequentialSampler(data_source=dataset)
        print("📊 Using SequentialSampler")

    return sampler


if __name__ == "__main__":
    main()
