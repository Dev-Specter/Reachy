#!/usr/bin/env python3

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.callbacks import BaseCallback


class DynamicsModel(nn.Module):
    """
    One member of the MBPO dynamics ensemble.

    Input:
        observation + action

    Predicts:
        delta observation
        reward
        terminal probability
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=128,
    ):
        super().__init__()

        input_dim = obs_dim + action_dim

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.delta_head = nn.Linear(
            hidden_dim,
            obs_dim,
        )

        self.reward_head = nn.Linear(
            hidden_dim,
            1,
        )

        self.done_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(self, x):

        features = self.backbone(x)

        delta_obs = self.delta_head(
            features
        )

        reward_scaled = self.reward_head(
            features
        )

        done_logit = self.done_head(
            features
        )

        return (
            delta_obs,
            reward_scaled,
            done_logit,
        )


class MBPOCallback(BaseCallback):
    """
    Adds the model-based component to SAC.

    Real transitions are collected from Gazebo.

    Periodically:
        1. train an ensemble of dynamics models
        2. sample real states
        3. generate short synthetic transitions
        4. add synthetic transitions to SAC replay buffer
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        ensemble_size=5,
        model_warmup_steps=1000,
        model_train_frequency=500,
        model_gradient_steps=50,
        model_batch_size=256,
        synthetic_transitions=1000,
        real_buffer_size=50000,
        reward_scale=30.0,
        seed=42,
        verbose=1,
    ):
        super().__init__(verbose)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.ensemble_size = (
            ensemble_size
        )

        self.model_warmup_steps = (
            model_warmup_steps
        )

        self.model_train_frequency = (
            model_train_frequency
        )

        self.model_gradient_steps = (
            model_gradient_steps
        )

        self.model_batch_size = (
            model_batch_size
        )

        self.synthetic_transitions = (
            synthetic_transitions
        )

        self.reward_scale = (
            reward_scale
        )

        self.rng = np.random.default_rng(
            seed
        )

        torch.manual_seed(seed)

        self.device = torch.device(
            "cpu"
        )

        self.real_buffer = deque(
            maxlen=real_buffer_size
        )

        self.models = []

        self.optimizers = []

        for member in range(
            ensemble_size
        ):

            torch.manual_seed(
                seed + member
            )

            model = DynamicsModel(
                obs_dim=obs_dim,
                action_dim=action_dim,
            ).to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-3,
            )

            self.models.append(model)
            self.optimizers.append(
                optimizer
            )

        self.previous_obs = None

        self.total_synthetic = 0

    def _on_training_start(self):

        if self.model._last_obs is None:
            raise RuntimeError(
                "MBPO could not obtain "
                "initial observation."
            )

        self.previous_obs = (
            np.asarray(
                self.model._last_obs[0],
                dtype=np.float32,
            ).copy()
        )

        print(
            "\nMBPO dynamics ensemble ready."
        )

        print(
            f"Ensemble models: "
            f"{self.ensemble_size}"
        )

        print(
            f"Model warmup: "
            f"{self.model_warmup_steps} "
            f"real steps"
        )

        print(
            "Synthetic rollout horizon: 1"
        )

    def _store_real_transition(self):

        actions = np.asarray(
            self.locals["actions"],
            dtype=np.float32,
        )

        new_obs = np.asarray(
            self.locals["new_obs"],
            dtype=np.float32,
        )

        rewards = np.asarray(
            self.locals["rewards"],
            dtype=np.float32,
        )

        dones = np.asarray(
            self.locals["dones"]
        )

        infos = self.locals["infos"]

        info = infos[0]

        if (
            bool(dones[0])
            and "terminal_observation"
            in info
        ):
            next_obs = np.asarray(
                info["terminal_observation"],
                dtype=np.float32,
            ).copy()

        else:
            next_obs = (
                new_obs[0].copy()
            )

        reason = info.get(
            "termination_reason",
            "",
        )

        # True environment terminal events.
        # Timeout is excluded because it is an
        # artificial episode horizon rather than
        # a physical terminal state.
        model_done = (
            reason == "goal"
            or reason == "fall"
        )

        transition = (
            self.previous_obs.copy(),
            actions[0].copy(),
            float(rewards[0]),
            next_obs,
            float(model_done),
        )

        self.real_buffer.append(
            transition
        )

        # new_obs already contains the reset
        # observation when VecEnv automatically
        # resets after an episode.
        self.previous_obs = (
            new_obs[0].copy()
        )

    def _prepare_dataset(self):

        data = list(
            self.real_buffer
        )

        observations = np.stack(
            [item[0] for item in data]
        ).astype(np.float32)

        actions = np.stack(
            [item[1] for item in data]
        ).astype(np.float32)

        rewards = np.asarray(
            [item[2] for item in data],
            dtype=np.float32,
        ).reshape(-1, 1)

        next_observations = np.stack(
            [item[3] for item in data]
        ).astype(np.float32)

        dones = np.asarray(
            [item[4] for item in data],
            dtype=np.float32,
        ).reshape(-1, 1)

        delta_observations = (
            next_observations
            - observations
        )

        inputs = np.concatenate(
            [
                observations,
                actions,
            ],
            axis=1,
        )

        reward_targets = (
            rewards / self.reward_scale
        )

        return (
            inputs,
            delta_observations,
            reward_targets,
            dones,
        )

    def _train_dynamics_models(self):

        (
            inputs_np,
            delta_np,
            reward_np,
            done_np,
        ) = self._prepare_dataset()

        num_samples = len(
            inputs_np
        )

        inputs = torch.as_tensor(
            inputs_np,
            dtype=torch.float32,
            device=self.device,
        )

        delta_targets = (
            torch.as_tensor(
                delta_np,
                dtype=torch.float32,
                device=self.device,
            )
        )

        reward_targets = (
            torch.as_tensor(
                reward_np,
                dtype=torch.float32,
                device=self.device,
            )
        )

        done_targets = (
            torch.as_tensor(
                done_np,
                dtype=torch.float32,
                device=self.device,
            )
        )

        validation_size = max(
            1,
            int(
                0.20 * num_samples
            ),
        )

        permutation = (
            self.rng.permutation(
                num_samples
            )
        )

        validation_indices = (
            permutation[
                :validation_size
            ]
        )

        training_indices = (
            permutation[
                validation_size:
            ]
        )

        if len(training_indices) == 0:
            training_indices = (
                validation_indices
            )

        validation_tensor = (
            torch.as_tensor(
                validation_indices,
                dtype=torch.long,
                device=self.device,
            )
        )

        validation_losses = []

        for (
            model,
            optimizer,
        ) in zip(
            self.models,
            self.optimizers,
        ):

            model.train()

            for _ in range(
                self.model_gradient_steps
            ):

                batch_size = min(
                    self.model_batch_size,
                    len(
                        training_indices
                    ),
                )

                batch_indices = (
                    self.rng.choice(
                        training_indices,
                        size=batch_size,
                        replace=(
                            len(
                                training_indices
                            )
                            < batch_size
                        ),
                    )
                )

                batch_tensor = (
                    torch.as_tensor(
                        batch_indices,
                        dtype=torch.long,
                        device=self.device,
                    )
                )

                (
                    delta_prediction,
                    reward_prediction,
                    done_logit,
                ) = model(
                    inputs[batch_tensor]
                )

                delta_loss = F.mse_loss(
                    delta_prediction,
                    delta_targets[
                        batch_tensor
                    ],
                )

                reward_loss = (
                    F.smooth_l1_loss(
                        reward_prediction,
                        reward_targets[
                            batch_tensor
                        ],
                    )
                )

                done_loss = (
                    F.binary_cross_entropy_with_logits(
                        done_logit,
                        done_targets[
                            batch_tensor
                        ],
                    )
                )

                loss = (
                    delta_loss
                    + reward_loss
                    + 0.1 * done_loss
                )

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=10.0,
                )

                optimizer.step()

            model.eval()

            with torch.no_grad():

                (
                    delta_prediction,
                    reward_prediction,
                    done_logit,
                ) = model(
                    inputs[
                        validation_tensor
                    ]
                )

                validation_delta = (
                    F.mse_loss(
                        delta_prediction,
                        delta_targets[
                            validation_tensor
                        ],
                    ).item()
                )

                validation_reward = (
                    F.smooth_l1_loss(
                        reward_prediction,
                        reward_targets[
                            validation_tensor
                        ],
                    ).item()
                )

                validation_done = (
                    F.binary_cross_entropy_with_logits(
                        done_logit,
                        done_targets[
                            validation_tensor
                        ],
                    ).item()
                )

                validation_losses.append(
                    (
                        validation_delta,
                        validation_reward,
                        validation_done,
                    )
                )

        losses = np.asarray(
            validation_losses,
            dtype=np.float32,
        )

        return (
            float(losses[:, 0].mean()),
            float(losses[:, 1].mean()),
            float(losses[:, 2].mean()),
        )

    def _predict_with_ensemble(
        self,
        observations,
        actions,
    ):

        batch_size = len(
            observations
        )

        next_observations = (
            np.zeros_like(
                observations,
                dtype=np.float32,
            )
        )

        rewards = np.zeros(
            batch_size,
            dtype=np.float32,
        )

        dones = np.zeros(
            batch_size,
            dtype=np.float32,
        )

        member_indices = (
            self.rng.integers(
                low=0,
                high=self.ensemble_size,
                size=batch_size,
            )
        )

        model_inputs = np.concatenate(
            [
                observations,
                actions,
            ],
            axis=1,
        ).astype(np.float32)

        for member in range(
            self.ensemble_size
        ):

            selection = np.where(
                member_indices == member
            )[0]

            if len(selection) == 0:
                continue

            x = torch.as_tensor(
                model_inputs[selection],
                dtype=torch.float32,
                device=self.device,
            )

            model = self.models[
                member
            ]

            model.eval()

            with torch.no_grad():

                (
                    delta_prediction,
                    reward_prediction,
                    done_logit,
                ) = model(x)

                delta_prediction = (
                    delta_prediction
                    .cpu()
                    .numpy()
                )

                reward_prediction = (
                    reward_prediction
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                done_probability = (
                    torch.sigmoid(
                        done_logit
                    )
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

            next_observations[
                selection
            ] = (
                observations[selection]
                + delta_prediction
            )

            rewards[
                selection
            ] = (
                reward_prediction
                * self.reward_scale
            )

            dones[
                selection
            ] = (
                done_probability
                >= 0.5
            ).astype(np.float32)

        return (
            next_observations,
            rewards,
            dones,
        )

    def _generate_synthetic_transitions(
        self
    ):

        data = list(
            self.real_buffer
        )

        num_real = len(data)

        if num_real == 0:
            return 0

        count = min(
            self.synthetic_transitions,
            num_real,
        )

        indices = self.rng.integers(
            low=0,
            high=num_real,
            size=count,
        )

        observations = np.stack(
            [
                data[index][0]
                for index in indices
            ]
        ).astype(np.float32)

        # Current SAC policy chooses actions
        # from real states.
        actions, _ = (
            self.model.predict(
                observations,
                deterministic=False,
            )
        )

        actions = np.asarray(
            actions,
            dtype=np.float32,
        )

        (
            next_observations,
            rewards,
            dones,
        ) = self._predict_with_ensemble(
            observations,
            actions,
        )

        # Synthetic samples are added to the
        # same replay buffer used by SAC.
        for index in range(count):

            self.model.replay_buffer.add(
                observations[
                    index:index + 1
                ],
                next_observations[
                    index:index + 1
                ],
                actions[
                    index:index + 1
                ],
                np.asarray(
                    [rewards[index]],
                    dtype=np.float32,
                ),
                np.asarray(
                    [dones[index]],
                    dtype=np.float32,
                ),
                [
                    {
                        "TimeLimit.truncated":
                        False
                    }
                ],
            )

        self.total_synthetic += count

        return count

    def _on_step(self):

        self._store_real_transition()

        timestep = self.num_timesteps

        should_train_model = (
            timestep
            >= self.model_warmup_steps
            and timestep
            % self.model_train_frequency
            == 0
        )

        if should_train_model:

            (
                delta_loss,
                reward_loss,
                done_loss,
            ) = (
                self._train_dynamics_models()
            )

            synthetic_added = (
                self._generate_synthetic_transitions()
            )

            print(
                "\n[MBPO] "
                f"real_steps={timestep} | "
                f"real_buffer="
                f"{len(self.real_buffer)} | "
                f"synthetic_added="
                f"{synthetic_added} | "
                f"synthetic_total="
                f"{self.total_synthetic}"
            )

            print(
                "[MBPO] validation | "
                f"delta_mse="
                f"{delta_loss:.6f} | "
                f"reward_loss="
                f"{reward_loss:.6f} | "
                f"done_loss="
                f"{done_loss:.6f}"
            )

        return True

    def save_dynamics(
        self,
        path,
    ):

        checkpoint = {
            "obs_dim":
            self.obs_dim,

            "action_dim":
            self.action_dim,

            "ensemble_size":
            self.ensemble_size,

            "reward_scale":
            self.reward_scale,

            "total_synthetic":
            self.total_synthetic,

            "models":
            [
                model.state_dict()
                for model
                in self.models
            ],
        }

        torch.save(
            checkpoint,
            path,
        )
