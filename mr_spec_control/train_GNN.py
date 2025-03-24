
import os
from pathlib import Path

import torch_geometric
from torch import nn
import torch
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig, GnnConfig, SequenceModelConfig
from environments.custom_vmas.common import CustomVmasTask

import wandb

wandb.login()

if __name__ == "__main__":
    intermediate_dims = [[32,32,32],[16,16,16],[8,8,8],[32,16,8]]

    for i, dims in enumerate(intermediate_dims):
        # Hyperparameters
        train_device = "cuda" # @param {"type":"string"}
        vmas_device = "cuda" # @param {"type":"string"}

        # Load task configuration
        task_config_path = "mr_spec_control/conf/task/custom_vmas/discovery_mothership_GNN.yaml"
        task = CustomVmasTask.DISCOVERY_OBSTACLES.get_from_yaml(task_config_path)

        # Load RL algorithm config
        # Loads from "benchmarl/conf/algorithm/mappo.yaml"
        algorithm_config = IppoConfig.get_from_yaml()
        algorithm_config = IppoConfig(
            share_param_critic=True, # Critic param sharing on
            clip_epsilon=0.2,
            entropy_coef=0.001, # We modify this, default is 0
            critic_coef=1,
            loss_critic_type="l2",
            lmbda=0.9,
            scale_mapping="biased_softplus_1.0", # Mapping for standard deviation
            use_tanh_normal=True,
            minibatch_advantage=True,
        )

        model_config = SequenceModelConfig(
            model_configs=[
                GnnConfig(
                                topology="full",
                                self_loops=False,
                                gnn_class=torch_geometric.nn.conv.TransformerConv,
                                gnn_kwargs={},
                            ),
                GnnConfig(
                                topology="full",
                                self_loops=False,
                                gnn_class=torch_geometric.nn.conv.TransformerConv,
                                gnn_kwargs={},
                            ),
                GnnConfig(
                                topology="full",
                                self_loops=False,
                                gnn_class=torch_geometric.nn.conv.TransformerConv,
                                gnn_kwargs={},
                            ),
                MlpConfig(
                    num_cells=[8],
                    activation_class=torch.nn.Tanh,
                    layer_class=torch.nn.Linear,
                ),
            ],
            intermediate_sizes=dims

        )

        # Load experiment configuration
        experiment_config = ExperimentConfig.get_from_yaml()

        # Override config parameters
        experiment_config.sampling_device = vmas_device
        experiment_config.train_device = train_device

        experiment_config.lr = 1e-5 # 0.00005 og val
        experiment_config.gamma = 0.99 # 0.99
        experiment_config.clip_grad_norm = True
        # experiment_config.clip_grad_val = 0.5

        experiment_config.max_n_frames = 1_024_000 # Number of frames before training ends
        experiment_config.on_policy_collected_frames_per_batch = 1_024 # Number of frames collected each iteration (max_steps from config * n_envs_per_worker)
        experiment_config.on_policy_n_envs_per_worker = 16 # Number of vmas vectorized enviornemnts (each will collect up to max_steps steps, see max_steps in task_config -> 50 * max_steps = 5_000 the number above)
        experiment_config.on_policy_n_minibatch_iters = 32
        experiment_config.on_policy_minibatch_size = 256
        experiment_config.keep_checkpoints_num = None

        experiment_config.evaluation = True
        experiment_config.render = True
        experiment_config.share_policy_params = False # Policy parameter sharing
        experiment_config.evaluation_interval = 5*experiment_config.on_policy_collected_frames_per_batch
        # experiment_config.evaluation_interval = 12_000 # Interval in terms of frames, will evaluate every eval_interval/frames_per_batch = 5 iterations
        experiment_config.evaluation_episodes = 10 # Number of vmas vectorized enviornemnts used in evaluation

        experiment_config.save_folder = f"runs/{i}_coord" # Folder where the experiment will be saved
        experiment_config.checkpoint_interval = 10*1024

        experiment_config.loggers = ["wandb"] # Log to csv, usually you should use wandb
        experiment_config.project_name = "mr_spec_control" # Wandb project name

        experiment = Experiment(
                    algorithm_config=algorithm_config,
                    model_config=model_config,
                    seed=0,
                    config=experiment_config,
                    task=task,
                )

        # Run experiment
        experiment.run()
