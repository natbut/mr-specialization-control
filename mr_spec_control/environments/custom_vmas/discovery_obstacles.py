#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator import rendering
from vmas.simulator.core import (Action, Agent, Box, Entity, Landmark, Sphere,
                                 World)
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar, Sensor
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

try:
    from environments.custom_vmas.camera import TopDownCamera
except:
    print("Checking local folder for camera")
    from camera import TopDownCamera

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.use_mothership = kwargs.pop("use_mothership", False)
        self.n_agents = kwargs.pop("n_agents", 3)
        if self.use_mothership: # mothership adds extra agent
            self.n_agents += 1
        self.n_targets = kwargs.pop("n_targets", 10)
        self.n_obstacles = kwargs.pop("n_obstacles", 10)
        self.x_semidim = kwargs.pop("x_semidim", 1)
        self.y_semidim = kwargs.pop("y_semidim", 1)
        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.25)
        self._covering_range = kwargs.pop("covering_range", 0.15)

        self.use_gnn = kwargs.pop("use_gnn", False)
        self.use_camera = kwargs.pop("use_camera", True)
        self.use_target_lidar = kwargs.pop("use_target_lidar", False)
        self.use_agent_lidar = kwargs.pop("use_agent_lidar", False)
        self.use_obstacle_lidar = kwargs.pop("use_obstacle_lidar", False)
        self.frame_x_dim = kwargs.pop("frame_x_dim", 2.0)
        self.frame_y_dim = kwargs.pop("frame_y_dim", 2.0)
        self._lidar_range = kwargs.pop("lidar_range", 0.25)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 32)

        self._agents_per_target = kwargs.pop("agents_per_target", 1)
        self.targets_respawn = kwargs.pop("targets_respawn", False)
        self.shared_reward = kwargs.pop("shared_reward", False)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -0.1)
        self.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 25.0)
        self.approach_rew_coeff = kwargs.pop("approach_rew_coeff", 100.0)
        self.time_penalty = kwargs.pop("time_penalty", 0.0)

        self._comms_range = kwargs.pop("comms_range", 1.5*self._lidar_range)
        self.min_collision_distance = kwargs.pop("min_collision_distance", 0.05)
        self.agent_radius = kwargs.pop("agent_radius", 0.025)

        self.target_radius = 2*self.agent_radius
        self.viewer_zoom = 1
        self.target_color = Color.GREEN
        self.step = 0

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            collision_force=500,
            substeps=2,
            drag=0.25,
        )

        # Lidar filters
        entity_filter_agents: Callable[[Entity], bool] = lambda e: e.name.startswith("agent")

        entity_filter_targets: Callable[[Entity], bool] = lambda e: e.name.startswith("target")

        entity_filter_obstacles: Callable[[Entity], bool] = lambda e: e.name.startswith("obstacle")

        # Initialize coordinator
        # mothership =
        # mothership.collision_rew = torch.zeros(batch_dim, device=device)
        # mothership.covering_reward = mothership.collision_rew.clone()
        # world.add_agent(mothership)

        # Initialize passengers
        for i in range(self.n_agents):
            # Mothership
            if i == 0 and self.use_mothership:
                name = f"mothership_{i}"
                movable = False
            else:
                name = f"passenger_{i}"
                movable = True

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=name,
                collide=True,
                shape=Sphere(radius=self.agent_radius),
                mass=5,
                max_speed=5.0,
                u_multiplier=2.0, #4.0,
                movable=movable,
                sensors=(
                    (
                        [
                        TopDownCamera(
                            world,
                            frame_x_dim=self.frame_x_dim,
                            frame_y_dim=self.frame_y_dim,
                            center=None
                            )
                        ]
                        if self.use_camera
                            else []
                    )
                    + (
                        [
                        Lidar(
                            world,
                            n_rays=self.n_lidar_rays,
                            max_range=self._lidar_range,
                            entity_filter=entity_filter_targets,
                            render_color=Color.GREEN,
                            )
                        ]
                        if self.use_target_lidar
                            else []
                    )
                    + (
                        [
                            Lidar(
                                world,
                                # angle_start=0.05,
                                # angle_end=2 * torch.pi + 0.05,
                                n_rays=self.n_lidar_rays,
                                max_range=self._lidar_range,
                                entity_filter=entity_filter_agents,
                                render_color=Color.BLUE,
                            )
                        ]
                        if self.use_agent_lidar
                        else []
                    )
                    + (
                        [
                            Lidar(
                                world,
                                # angle_start=0.1,
                                # angle_end=2 * torch.pi + 0.1,
                                n_rays=self.n_lidar_rays,
                                max_range=self._lidar_range,
                                entity_filter=entity_filter_obstacles,
                                render_color=Color.RED,
                            )
                        ]
                        if self.use_obstacle_lidar
                        else []
                    )
                ),
            )
            agent.collision_rew = torch.zeros(batch_dim, device=device)
            agent.covering_reward = torch.zeros(batch_dim, device=device)
            agent.task_dist = torch.zeros(batch_dim, device=device)
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)

        self._targets = []
        for i in range(self.n_targets):
            target = Landmark(
                name=f"target_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=self.target_radius),
                color=self.target_color,
            )
            world.add_landmark(target)
            self._targets.append(target)

        self.covered_targets = torch.zeros(batch_dim, self.n_targets, device=device)
        self.shared_covering_rew = torch.zeros(batch_dim, device=device)
        self.time_rew = torch.zeros(batch_dim, device=device)

        self._obstacles = []
        for i in range(self.n_obstacles):
            length = torch.rand(1).item()*0.25 + 0.1
            width = torch.rand(1).item()*0.25 + 0.1
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Box(length=length, width=width),
                color=Color.GRAY,
            )
            world.add_landmark(obstacle)
            self._obstacles.append(obstacle)

        return world

    def reset_world_at(self, env_index: int = None):
        """Reset world at the given environment index"""
        self.step = 0
        # First-time startup
        if env_index is None:
            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_targets),
                False,
                device=self.world.device,
            )
        else:
            self.all_time_covered_targets[env_index] = False

        # Place entities
        placable_entities = self._obstacles[: self.n_obstacles] + \
            self._targets[: self.n_targets] + \
                self.world.agents[: self.n_agents]

        ScenarioUtils.spawn_entities_randomly(
            entities=placable_entities,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=self._min_dist_between_entities,
            x_bounds=(-self.world.x_semidim, self.world.x_semidim),
            y_bounds=(-self.world.y_semidim, self.world.y_semidim),
        )
        for target in self._targets[self.n_targets :]:
            target.set_pos(self.get_outside_pos(env_index), batch_index=env_index)

        for agent in self.world.agents:
            # agent.set_pos(
            #         torch.tensor(
            #             [0.0, 0.0],
            #             dtype=torch.float32,
            #             device=self.world.device,
            #         ),
            #         batch_index=env_index,
            #     )

            dists_to_tasks = torch.stack([
            torch.linalg.norm(agent.state.pos - t.state.pos, dim=1) for t in self._targets
            ], dim=-1)
            nearest_task_dist = (torch.min(dists_to_tasks, dim=1).values)
            agent.task_dist = nearest_task_dist


            # if env_index is not None:
            #     agent.collision_rew[env_index] = 0
            #     agent.covering_reward[env_index] = 0
            #     agent.task_dist[env_index] = 0
            #     agent.pos_rew[env_index] = 0

        # Spawn passengers around mothership
        # mothership_pos = self.world.agents[0].state.pos
        # for agent in self.world.agents[1:]:
        #     agent.set_pos(mothership_pos + (torch.rand(1, 2, device=self.world.device)*2 - 1)*0.1, batch_index=env_index)


    def reward(self, agent: Agent):
        """Reward completing targets, avoiding collisions with agents, and time penalty"""

        # TODO Compute mothership reward
        if self.use_mothership:
            is_first = agent == self.world.agents[1] # 0 is mothership
        else:
            is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        # Time reward, Covering reward
        if is_first:
            self.time_rew = torch.full(
                (self.world.batch_dim,),
                self.time_penalty,
                device=self.world.device,
            )
            self.agents_pos = torch.stack(
                [a.state.pos for a in self.world.agents], dim=1
            )
            self.targets_pos = torch.stack([t.state.pos for t in self._targets], dim=1)
            self.agents_targets_dists = torch.cdist(self.agents_pos, self.targets_pos)
            self.agents_per_target = torch.sum(
                (self.agents_targets_dists < self._covering_range).type(torch.int),
                dim=1,
            )
            self.covered_targets = self.agents_per_target >= self._agents_per_target

            self.shared_covering_rew[:] = 0
            for a in self.world.agents:
                self.shared_covering_rew += self.passenger_covering_reward(a)
            self.shared_covering_rew[self.shared_covering_rew != 0] /= 2

        covering_rew = (
            agent.covering_reward
            if not self.shared_reward
            else self.shared_covering_rew
        )

        # Respawn targets
        if is_last:
            if self.targets_respawn:
                occupied_positions_agents = [self.agents_pos]
                for i, target in enumerate(self._targets):
                    occupied_positions_targets = [
                        o.state.pos.unsqueeze(1)
                        for o in self._targets
                        if o is not target
                    ]
                    occupied_positions = torch.cat(
                        occupied_positions_agents + occupied_positions_targets,
                        dim=1,
                    )
                    pos = ScenarioUtils.find_random_pos_for_entity(
                        occupied_positions,
                        env_index=None,
                        world=self.world,
                        min_dist_between_entities=self._min_dist_between_entities,
                        x_bounds=(-self.world.x_semidim, self.world.x_semidim),
                        y_bounds=(-self.world.y_semidim, self.world.y_semidim),
                    )

                    target.state.pos[self.covered_targets[:, i]] = pos[
                        self.covered_targets[:, i]
                    ].squeeze(1)
            else:
                self.all_time_covered_targets += self.covered_targets
                for i, target in enumerate(self._targets):
                    target.state.pos[self.covered_targets[:, i]] = self.get_outside_pos(
                        None
                    )[self.covered_targets[:, i]]

        # Collision avoidance reward
        agent.collision_rew[:] = 0
        for o in self._obstacles:
            agent.collision_rew[self.world.get_distance(o, agent) <
                                self.min_collision_distance] += self.agent_collision_penalty

        # Distance to nearest task reward
        agent.pos_rew[:] = 0
        dists_to_tasks = torch.stack([
            torch.linalg.norm(agent.state.pos - t.state.pos, dim=1) for t in self._targets
        ], dim=-1)
        nearest_task_dist = (torch.min(dists_to_tasks, dim=1).values)
        # print(f"\n\n!! Dists to tasks: {dists_to_tasks}, \n !! Nearest task dist: {nearest_task_dist}")

        agent.pos_rew[:] = (agent.task_dist - nearest_task_dist) * self.approach_rew_coeff
        agent.pos_rew[covering_rew > 0] = 0
        agent.task_dist = nearest_task_dist

        # pos_rew[covering_rew > 0] = 0 # No approaching task reward if at task
        self.step += 1
        # print(f"\n!!{self.step} Rewards: \n\tCollision: {agent.collision_rew}\n\tCovering: {covering_rew}\n\tTime: {self.time_rew}\n\tPos: {agent.pos_rew}")

        return agent.collision_rew + covering_rew + self.time_rew + agent.pos_rew

    def get_outside_pos(self, env_index):
        return torch.empty(
            (
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p)
            ),
            device=self.world.device,
        ).uniform_(-1000 * self.world.x_semidim, -10 * self.world.x_semidim)

    def passenger_covering_reward(self, agent):
        """Reward for covering targets"""
        agent_index = self.world.agents.index(agent)

        agent.covering_reward[:] = 0
        targets_covered_by_agent = (
            self.agents_targets_dists[:, agent_index] < self._covering_range
        )
        num_covered_targets_covered_by_agent = (
            targets_covered_by_agent * self.covered_targets
        ).sum(dim=-1)
        agent.covering_reward += (
            num_covered_targets_covered_by_agent * self.covering_rew_coeff
        )
        return agent.covering_reward

    def observation(self, agent: Agent):
        """
        The returned tensor should contain the observations for ``agent`` in all envs and should have
        shape ``(self.world.batch_dim, n_agent_obs)``, or be a dict with leaves following that shape.
        """
        if self.use_gnn:
            agent_poses = []
            target_poses = []
            obstacle_poses = []

            radius = 1.0

            for a in self.world.agents:
                relative_pos = agent.state.pos - a.state.pos
                a_dists = torch.linalg.norm(relative_pos, dim=1)
                # relative_pos[a_dists > radius] = 0
                agent_poses.append(relative_pos)

            for t in self._targets:
                relative_pos = agent.state.pos - t.state.pos
                t_dists = torch.linalg.norm(relative_pos, dim=1)
                # relative_pos[t_dists > radius] = 0
                target_poses.append(relative_pos)

            for o in self._obstacles:
                relative_pos = agent.state.pos - o.state.pos
                o_dists = torch.linalg.norm(relative_pos, dim=1)
                # relative_pos[o_dists > radius] = 0
                obstacle_poses.append(relative_pos)


            obs = torch.cat(
                            [
                                agent.state.pos,
                                agent.state.vel,
                            ]
                            + agent_poses
                            + target_poses
                            + obstacle_poses,
                            dim=-1,
                        )

            # print("Agent info:", [agent.state.pos, agent.state.vel])
            # print("Agent poses", agent_poses)
            # print("Target poses:", target_poses)
            # print("Obstacle poses:", obstacle_poses)

            # print("OBS: ", obs)

            return obs

            # Agent position
            # agent_position = agent.state.pos

            # # Tasks positions
            # task0_position = self._targets[0].state.pos

            # obs = {
            #     "position_key": agent_position,  # Used to build the dynamic graph
            #     "task0_pos": task0_position,  # Modify if other features are needed
            # }

            # # print("OBS: ", obs)

            # return obs


        obs = {}
        # NOTE Passengers get ONLY their sensor views
        if "mothership" in agent.name:
            # Mothership obs (global agents & tasks)
            obs["passenger_pos"] = torch.cat(
                    [a.state.pos for a in self.world.agents[1:]], dim=1
                )
            obs["target_pos"] = torch.cat([t.state.pos for t in self._targets], dim=1)
        else:
            # passenger obs (local lidar scans + mothership guidance)
            if self.use_camera:
                obs["camera"] = agent.sensors[0].measure(agent.state.pos) / 255
            if self.use_target_lidar:
                obs["target_lidar"] = agent.sensors[1].measure()
            if self.use_agent_lidar:
                obs["agent_lidar"] = agent.sensors[2].measure()
            if self.use_obstacle_lidar:
                obs["obstacle_lidar"] = agent.sensors[3].measure()

            # TODO: Extract passenger-specific actions from mothership.action.u
            if self.use_mothership:
                if self.world.agents[0].action.u is None:
                    obs["mothership_actions"] = torch.zeros((self.world.batch_dim,
                                                            self.world.agents[0].action_size),
                                                            device=self.world.device)
                else:
                    obs["mothership_actions"] = self.world.agents[0].action.u

        return obs

    # def get_gnn_observation_full(self, agent: Agent):
    #     """Generate GNN observations with full topology."""

    #     # Node feature matrix: Include position and other agent-specific features
    #     node_features = []

    #     # Agent nodes
    #     for agent in self.world.agents:
    #         node_features.append(agent.state.pos)

    #     # Obstacle nodes (only position is needed)
    #     for obst in self._obstacles:
    #         node_features.append(obst.state.pos)

    #     # Target nodes (only position is needed)
    #     for target in self._targets:
    #         node_features.append(target.state.pos)

    #     node_features = torch.stack(node_features).to(self.world.device)

    #     # Edge index: Fully connected graph
    #     edge_index = []
    #     edge_features = []
    #     for i in range(self.n_agents + self.n_obstacles + self.n_targets):
    #         for j in range(self.n_agents + self.n_obstacles + self.n_targets):
    #             if i != j:  # No self-loops
    #                 edge_index.append([i, j])

    #                 distance = torch.norm(node_features[i, :2] - node_features[j, :2], p=2).unsqueeze(0)
    #                 edge_features.append(distance)

    #     # Convert to tensors for PyTorch Geometric
    #     # node_features = torch.tensor(node_features, device=self.world.device)
    #     edge_index = torch.tensor(edge_index, device=self.world.device).T  # Shape (2, num_edges)
    #     edge_features = torch.stack(edge_features).to(self.world.device)  # Shape: (num_edges, 1)

    #     return node_features, edge_index, edge_features

    # def get_gnn_obs_fromPos(self, agent: Agent):
    #     obs = {}

    #     obs["position_key"] = agent.state.pos
    #     obs["edge_index"]

    #     return obs


    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {
            "covering_reward": (
                agent.covering_reward
                if not self.shared_reward
                else self.shared_covering_rew
            ),
            "collision_rew": agent.collision_rew,
            "targets_covered": self.covered_targets.sum(-1),
        }
        return info

    def done(self):
        done = self.all_time_covered_targets.all(dim=-1)
        return done

    def extra_render(self, env_index: int = 0) -> "List[Geom]":

        geoms: List[Geom] = []
        # Target ranges
        for target in self._targets:
            range_circle = rendering.make_circle(self._covering_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*self.target_color.value)
            geoms.append(range_circle)
        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self._comms_range:
                    color = Color.RED.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms


if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(scenario)
