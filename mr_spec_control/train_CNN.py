
import os
from pathlib import Path

from benchmarl.algorithms import IppoConfig, MappoConfig, MaddpgConfig, IddpgConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import EnsembleModelConfig, MlpConfig, CnnConfig
from environments.custom_vmas.common import CustomVmasTask
from models.joint_models import JointModelsConfig

import wandb

wandb.login()

if __name__ == "__main__":

    # Hyperparameters
    train_device = "cuda" # @param {"type":"string"}
    vmas_device = "cuda" # @param {"type":"string"}

    # Load task configuration
    task_config_path = "mr_spec_control/conf/task/custom_vmas/discovery_mothership_CNN.yaml"
    task = CustomVmasTask.DISCOVERY_OBSTACLES.get_from_yaml(task_config_path)

    # Load RL algorithm config
    # Loads from "benchmarl/conf/algorithm/mappo.yaml"
    # algorithm_config = EnsembleAlgorithmConfig(
    #     {"mothership": MappoConfig.get_from_yaml(),
    #      "passenger": MappoConfig.get_from_yaml()
    #     }
    # )
    # algorithm_config = MappoConfig.get_from_yaml()
    algorithm_config = MappoConfig(
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

    # Load policy model configs
    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    cnn_config_path = "mr_spec_control/conf/models/custom_cnn.yaml"
    model_config = CnnConfig.get_from_yaml(cnn_config_path)
    # model_config = MlpConfig.get_from_yaml()

    # TODO Could try sequence model where CNN is shared by actor and critic

    critic_model_config = CnnConfig.get_from_yaml(cnn_config_path)
    # critic_model_config = MlpConfig.get_from_yaml()
    # critic_model_config.centralised = True
    # critic_model_config.has_agent_input_dim = True

    # Load experiment configuration
    experiment_config = ExperimentConfig.get_from_yaml()

    # Override config parameters
    experiment_config.sampling_device = vmas_device
    experiment_config.train_device = train_device

    # NOTE Changed lr and minibnatch size here
    # NOTE probably change to a 3-layer CNN too

    experiment_config.lr = 5e-5 # 0.00005
    experiment_config.gamma = 0.99 # 0.99
    experiment_config.clip_grad_norm = True
    # experiment_config.clip_grad_val = 0.5

    experiment_config.max_n_frames = 1_024_000 # Number of frames before training ends

    experiment_config.on_policy_collected_frames_per_batch = 1_024 # Number of frames collected each iteration (max_steps from config * n_envs_per_worker)
    experiment_config.on_policy_n_envs_per_worker = 16 # Number of vmas vectorized enviornemnts (each will collect up to max_steps steps, see max_steps in task_config -> 50 * max_steps = 5_000 the number above)
    experiment_config.on_policy_n_minibatch_iters = 3 #45
    experiment_config.on_policy_minibatch_size = 128 # 400 NOTE try 128?

    experiment_config.keep_checkpoints_num = 5
    experiment_config.save_folder = "runs" # Folder where the experiment will be saved
    experiment_config.checkpoint_interval = 1024

    experiment_config.evaluation = True
    experiment_config.render = False
    experiment_config.share_policy_params = True # Policy parameter sharing
    experiment_config.evaluation_interval = 10*experiment_config.on_policy_collected_frames_per_batch
    # experiment_config.evaluation_interval = 12_000 # Interval in terms of frames, will evaluate every eval_interval/frames_per_batch = 5 iterations
    experiment_config.evaluation_episodes = 8 # Number of vmas vectorized enviornemnts used in evaluation

    experiment_config.loggers = ["wandb"] # Log to csv, usually you should use wandb
    experiment_config.project_name = "mr_spec_control" # Wandb project name
    experiment_config.group_name = "channels" # Wandb group name
    experiment_config.run_name = "cnn_888" # Wandb run name

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )


    # Run experiment
    experiment.run()
