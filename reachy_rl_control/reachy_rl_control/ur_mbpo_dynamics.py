#!/usr/bin/env python3

import numpy as np
import torch

from reachy_rl_control.mbpo_dynamics import (
    MBPOCallback,
)


class URMBPOCallback(MBPOCallback):
    """
    Uncertainty- and Risk-Aware MBPO callback.

    This class extends the existing MBPOCallback without
    modifying the frozen baseline implementation.

    UR-MBPO adds:

    1. Ensemble disagreement estimation for every candidate
       synthetic transition.

    2. Rejection of synthetic transitions when ensemble
       disagreement is above an uncertainty threshold.

    3. Conservative reward penalties based on model
       uncertainty.

    4. Human-risk penalties using the predicted LiDAR
       portion of the next observation.

    5. Logging of uncertainty/risk interventions for later
       analysis.

    Human-aware observation layout
    -------------------------------

    Observation dimension = 20

    Indices:

        0   x
        1   y
        2   yaw
        3   vx
        4   vy
        5   wz
        6   goal_dx
        7   goal_dy

        8-19 normalized LiDAR sectors

    LiDAR normalization:

        0.0 = extremely close
        1.0 = 3 metres or farther
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

        # =================================================
        # UR-MBPO PARAMETERS
        # =================================================

        uncertainty_threshold=0.10,

        uncertainty_penalty_scale=2.0,

        safe_clearance=0.75,

        near_miss_clearance=0.50,

        lidar_max_distance=3.0,

        risk_penalty_scale=2.0,

        lidar_start_index=8,
    ):

        # =================================================
        # ORIGINAL MBPO INITIALISATION
        # =================================================

        super().__init__(
            obs_dim=obs_dim,

            action_dim=action_dim,

            ensemble_size=ensemble_size,

            model_warmup_steps=(
                model_warmup_steps
            ),

            model_train_frequency=(
                model_train_frequency
            ),

            model_gradient_steps=(
                model_gradient_steps
            ),

            model_batch_size=(
                model_batch_size
            ),

            synthetic_transitions=(
                synthetic_transitions
            ),

            real_buffer_size=(
                real_buffer_size
            ),

            reward_scale=reward_scale,

            seed=seed,

            verbose=verbose,
        )

        # =================================================
        # UR-MBPO CONFIGURATION
        # =================================================

        self.uncertainty_threshold = float(
            uncertainty_threshold
        )

        self.uncertainty_penalty_scale = float(
            uncertainty_penalty_scale
        )

        self.safe_clearance = float(
            safe_clearance
        )

        self.near_miss_clearance = float(
            near_miss_clearance
        )

        self.lidar_max_distance = float(
            lidar_max_distance
        )

        self.risk_penalty_scale = float(
            risk_penalty_scale
        )

        self.lidar_start_index = int(
            lidar_start_index
        )

        # -------------------------------------------------
        # CHECK HUMAN-AWARE OBSERVATION
        # -------------------------------------------------

        if (
            self.obs_dim
            <= self.lidar_start_index
        ):

            raise ValueError(
                "UR-MBPO expected LiDAR values "
                "inside the observation."
            )

        self.num_lidar_sectors = (
            self.obs_dim
            - self.lidar_start_index
        )

        # =================================================
        # UNCERTAINTY NORMALISATION
        # =================================================
        #
        # The observation dimensions have different units
        # and numerical scales.
        #
        # x/y are metres
        # yaw is radians
        # velocity values have smaller physical ranges
        # LiDAR values are already normalized [0, 1]
        #
        # Therefore raw ensemble standard deviation should
        # not be averaged directly across dimensions.
        # =================================================

        self.uncertainty_scales = np.ones(
            self.obs_dim,
            dtype=np.float32,
        )

        if self.obs_dim >= 8:

            self.uncertainty_scales[
                :8
            ] = np.array(
                [
                    2.0,       # x
                    2.0,       # y

                    np.pi,     # yaw

                    0.30,      # vx
                    0.30,      # vy
                    0.60,      # wz

                    0.50,      # goal_dx
                    0.50,      # goal_dy
                ],
                dtype=np.float32,
            )

        # LiDAR dimensions remain scale 1.0 because they
        # are already normalized to [0, 1].

        # =================================================
        # UR-MBPO STATISTICS
        # =================================================

        self.total_candidates = 0

        self.total_accepted = 0

        self.total_uncertainty_rejected = 0

        self.total_risk_penalized = 0

        self.total_near_miss_predictions = 0

        self.last_mean_uncertainty = float(
            "nan"
        )

        self.last_max_uncertainty = float(
            "nan"
        )

        self.last_min_uncertainty = float(
            "nan"
        )

        self.last_mean_predicted_clearance = float(
            "nan"
        )

        self.last_min_predicted_clearance = float(
            "nan"
        )

    # =====================================================
    # TRAINING START
    # =====================================================

    def _on_training_start(
        self,
    ):

        # Preserve the original MBPO startup behaviour.
        super()._on_training_start()

        print(
            "\n======================================"
        )

        print(
            "UR-MBPO SAFETY LAYER READY"
        )

        print(
            "======================================"
        )

        print(
            "Uncertainty threshold: "
            f"{self.uncertainty_threshold:.3f}"
        )

        print(
            "Uncertainty penalty scale: "
            f"{self.uncertainty_penalty_scale:.3f}"
        )

        print(
            "Safe LiDAR clearance: "
            f"{self.safe_clearance:.2f} m"
        )

        print(
            "Near-miss threshold: "
            f"{self.near_miss_clearance:.2f} m"
        )

        print(
            "LiDAR maximum distance: "
            f"{self.lidar_max_distance:.2f} m"
        )

        print(
            "Risk penalty scale: "
            f"{self.risk_penalty_scale:.3f}"
        )

        print(
            "LiDAR start index: "
            f"{self.lidar_start_index}"
        )

        print(
            "LiDAR sectors: "
            f"{self.num_lidar_sectors}"
        )

    # =====================================================
    # ENSEMBLE PREDICTION + UNCERTAINTY
    # =====================================================

    def _predict_ensemble_statistics(
        self,
        observations,
        actions,
    ):
        """
        Run every candidate transition through every
        dynamics ensemble member.

        Returns:

            next_observations
            rewards
            dones
            uncertainty
            predicted_clearance

        One ensemble member is randomly selected to provide
        the actual synthetic transition, preserving the same
        basic stochastic MBPO behaviour.

        Ensemble disagreement across all members is used to
        estimate epistemic model uncertainty.
        """

        observations = np.asarray(
            observations,
            dtype=np.float32,
        )

        actions = np.asarray(
            actions,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # MODEL INPUT
        # -------------------------------------------------

        model_inputs = np.concatenate(
            [
                observations,
                actions,
            ],
            axis=1,
        )

        input_tensor = torch.as_tensor(
            model_inputs,
            dtype=torch.float32,
            device=self.device,
        )

        next_observation_members = []

        reward_members = []

        done_probability_members = []

        # =================================================
        # RUN ALL ENSEMBLE MEMBERS
        # =================================================

        with torch.no_grad():

            for model in self.models:

                model.eval()

                (
                    delta_prediction,
                    reward_prediction,
                    done_logit,
                ) = model(
                    input_tensor
                )

                # -----------------------------------------
                # DELTA OBSERVATION
                # -----------------------------------------

                delta_np = (
                    delta_prediction
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32
                    )
                )

                next_obs_np = (
                    observations
                    +
                    delta_np
                )

                # -----------------------------------------
                # REWARD
                # -----------------------------------------

                reward_np = (
                    reward_prediction
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(
                        np.float32
                    )
                    *
                    self.reward_scale
                )

                # -----------------------------------------
                # TERMINATION PROBABILITY
                # -----------------------------------------

                done_probability_np = (
                    torch.sigmoid(
                        done_logit
                    )
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(
                        np.float32
                    )
                )

                next_observation_members.append(
                    next_obs_np
                )

                reward_members.append(
                    reward_np
                )

                done_probability_members.append(
                    done_probability_np
                )

        # -------------------------------------------------
        # SHAPES
        #
        # next_observation_members:
        #
        #   [ensemble, batch, obs_dim]
        #
        # reward_members:
        #
        #   [ensemble, batch]
        #
        # done_probability_members:
        #
        #   [ensemble, batch]
        # -------------------------------------------------

        next_observation_members = np.stack(
            next_observation_members,
            axis=0,
        ).astype(
            np.float32
        )

        reward_members = np.stack(
            reward_members,
            axis=0,
        ).astype(
            np.float32
        )

        done_probability_members = np.stack(
            done_probability_members,
            axis=0,
        ).astype(
            np.float32
        )

        # =================================================
        # ENSEMBLE UNCERTAINTY
        # =================================================
        #
        # Measure disagreement using the standard deviation
        # of predicted NEXT observations.
        #
        # Each observation dimension is normalized using
        # self.uncertainty_scales before aggregation.
        # =================================================

        ensemble_std = np.std(
            next_observation_members,
            axis=0,
        )

        normalized_std = (
            ensemble_std
            /
            self.uncertainty_scales[
                None,
                :
            ]
        )

        uncertainty = np.sqrt(
            np.mean(
                np.square(
                    normalized_std
                ),
                axis=1,
            )
        ).astype(
            np.float32
        )

        # =================================================
        # HUMAN RISK FROM ENSEMBLE-MEAN LiDAR
        # =================================================

        ensemble_mean_next_obs = np.mean(
            next_observation_members,
            axis=0,
        )

        predicted_lidar = (
            ensemble_mean_next_obs[
                :,
                self.lidar_start_index:
            ]
        )

        # The learned model may predict slightly outside
        # the valid normalized LiDAR range.
        predicted_lidar = np.clip(
            predicted_lidar,
            0.0,
            1.0,
        )

        # Minimum predicted normalized LiDAR value across
        # all sectors.
        minimum_normalized_lidar = np.min(
            predicted_lidar,
            axis=1,
        )

        # Convert normalized LiDAR back to metres.
        predicted_clearance = (
            minimum_normalized_lidar
            *
            self.lidar_max_distance
        ).astype(
            np.float32
        )

        # =================================================
        # SELECT ONE ENSEMBLE MEMBER
        # =================================================
        #
        # Standard MBPO-style model randomness:
        #
        # each candidate transition receives its next state,
        # reward and done estimate from one randomly chosen
        # dynamics model.
        # =================================================

        batch_size = (
            observations.shape[0]
        )

        chosen_members = self.rng.integers(
            low=0,
            high=self.ensemble_size,
            size=batch_size,
        )

        batch_indices = np.arange(
            batch_size
        )

        next_observations = (
            next_observation_members[
                chosen_members,
                batch_indices,
            ]
            .astype(
                np.float32
            )
        )

        rewards = (
            reward_members[
                chosen_members,
                batch_indices,
            ]
            .astype(
                np.float32
            )
        )

        done_probabilities = (
            done_probability_members[
                chosen_members,
                batch_indices,
            ]
        )

        dones = (
            done_probabilities
            >= 0.5
        ).astype(
            np.float32
        )

        return (
            next_observations,
            rewards,
            dones,
            uncertainty,
            predicted_clearance,
        )

    # =====================================================
    # SYNTHETIC TRANSITION GENERATION
    # =====================================================

    def _generate_synthetic_transitions(
        self,
    ):
        """
        Generate uncertainty- and risk-aware synthetic
        transitions.

        Procedure:

        1. Sample starting observations from the real
           Gazebo transition buffer.

        2. Ask the current SAC policy for actions.

        3. Predict next-state behaviour using all ensemble
           members.

        4. Measure ensemble disagreement.

        5. Reject predictions above the uncertainty
           threshold.

        6. Apply uncertainty and human-risk reward
           penalties to accepted synthetic transitions.

        7. Add accepted transitions to SAC's replay buffer.
        """

        data = list(
            self.real_buffer
        )

        num_real = len(
            data
        )

        if num_real == 0:

            return 0

        # Same candidate count as baseline MBPO.
        count = min(
            self.synthetic_transitions,
            num_real,
        )

        # =================================================
        # SAMPLE REAL STARTING STATES
        # =================================================

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
        ).astype(
            np.float32
        )

        # =================================================
        # CURRENT SAC POLICY SELECTS ACTIONS
        # =================================================

        actions, _ = self.model.predict(
            observations,
            deterministic=False,
        )

        actions = np.asarray(
            actions,
            dtype=np.float32,
        )

        # =================================================
        # ENSEMBLE PREDICTIONS
        # =================================================

        (
            next_observations,
            rewards,
            dones,
            uncertainty,
            predicted_clearance,
        ) = (
            self._predict_ensemble_statistics(
                observations,
                actions,
            )
        )

        # =================================================
        # UNCERTAINTY FILTER
        # =================================================
        #
        # Candidate synthetic transitions whose model
        # disagreement exceeds the threshold are discarded.
        # =================================================

        accepted_mask = (
            uncertainty
            <=
            self.uncertainty_threshold
        )

        uncertainty_rejected = int(
            np.sum(
                ~accepted_mask
            )
        )

        # =================================================
        # UNCERTAINTY PENALTY
        # =================================================
        #
        # Even accepted model transitions receive a small
        # conservative penalty proportional to uncertainty.
        # =================================================

        uncertainty_penalty = (
            self.uncertainty_penalty_scale
            *
            uncertainty
        )

        conservative_rewards = (
            rewards
            -
            uncertainty_penalty
        )

        # =================================================
        # HUMAN-RISK PENALTY
        # =================================================
        #
        # No synthetic risk penalty when predicted LiDAR
        # clearance >= SAFE_CLEARANCE.
        #
        # Below safe_clearance:
        #
        # risk_score =
        #
        #   (safe_clearance - predicted_clearance)
        #   --------------------------------------
        #              safe_clearance
        #
        # clipped into [0, 1].
        # =================================================

        risk_score = np.clip(
            (
                self.safe_clearance
                -
                predicted_clearance
            )
            /
            self.safe_clearance,

            0.0,
            1.0,
        ).astype(
            np.float32
        )

        risk_penalty = (
            self.risk_penalty_scale
            *
            risk_score
        )

        conservative_rewards = (
            conservative_rewards
            -
            risk_penalty
        )

        # -------------------------------------------------
        # ANALYSIS MASKS
        # -------------------------------------------------

        risk_penalized_mask = (
            risk_score
            > 0.0
        )

        near_miss_prediction_mask = (
            predicted_clearance
            <
            self.near_miss_clearance
        )

        # =================================================
        # ACCEPTED SYNTHETIC TRANSITIONS
        # =================================================

        accepted_indices = np.flatnonzero(
            accepted_mask
        )

        for index in accepted_indices:

            self.model.replay_buffer.add(

                # -----------------------------------------
                # CURRENT OBSERVATION
                # -----------------------------------------

                observations[
                    index:index + 1
                ],

                # -----------------------------------------
                # PREDICTED NEXT OBSERVATION
                # -----------------------------------------

                next_observations[
                    index:index + 1
                ],

                # -----------------------------------------
                # ACTION
                # -----------------------------------------

                actions[
                    index:index + 1
                ],

                # -----------------------------------------
                # CONSERVATIVE REWARD
                # -----------------------------------------

                np.asarray(
                    [
                        conservative_rewards[
                            index
                        ]
                    ],
                    dtype=np.float32,
                ),

                # -----------------------------------------
                # PREDICTED TERMINATION
                # -----------------------------------------

                np.asarray(
                    [
                        dones[
                            index
                        ]
                    ],
                    dtype=np.float32,
                ),

                # -----------------------------------------
                # INFO
                # -----------------------------------------

                [
                    {
                        "TimeLimit.truncated":
                            False
                    }
                ],
            )

        accepted_count = int(
            len(
                accepted_indices
            )
        )

        # =================================================
        # UPDATE STATISTICS
        # =================================================

        self.total_candidates += int(
            count
        )

        self.total_accepted += (
            accepted_count
        )

        self.total_uncertainty_rejected += (
            uncertainty_rejected
        )

        self.total_risk_penalized += int(
            np.sum(
                risk_penalized_mask
                &
                accepted_mask
            )
        )

        self.total_near_miss_predictions += int(
            np.sum(
                near_miss_prediction_mask
                &
                accepted_mask
            )
        )

        # -------------------------------------------------
        # KEEP BASELINE MBPO COUNTER SEMANTICS
        # -------------------------------------------------
        #
        # In baseline MBPO this counts transitions actually
        # inserted into the SAC replay buffer.
        # -------------------------------------------------

        self.total_synthetic += (
            accepted_count
        )

        # -------------------------------------------------
        # LATEST DIAGNOSTICS
        # -------------------------------------------------

        self.last_mean_uncertainty = float(
            np.mean(
                uncertainty
            )
        )

        self.last_max_uncertainty = float(
            np.max(
                uncertainty
            )
        )

        self.last_min_uncertainty = float(
            np.min(
                uncertainty
            )
        )

        self.last_mean_predicted_clearance = float(
            np.mean(
                predicted_clearance
            )
        )

        self.last_min_predicted_clearance = float(
            np.min(
                predicted_clearance
            )
        )

        return accepted_count

    # =====================================================
    # CALLBACK STEP
    # =====================================================

    def _on_step(
        self,
    ):
        """
        Preserve the baseline MBPO schedule:

            collect real transition every step

            after warmup:
                train dynamics every N steps
                generate synthetic transitions

        Only the synthetic-transition generation differs.
        """

        # -------------------------------------------------
        # STORE REAL GAZEBO TRANSITION
        # -------------------------------------------------

        self._store_real_transition()

        timestep = (
            self.num_timesteps
        )

        # -------------------------------------------------
        # ORIGINAL MBPO MODEL-TRAINING SCHEDULE
        # -------------------------------------------------

        should_train_model = (
            timestep
            >=
            self.model_warmup_steps

            and

            timestep
            %
            self.model_train_frequency
            ==
            0
        )

        if should_train_model:

            # =============================================
            # TRAIN DYNAMICS ENSEMBLE
            # =============================================

            (
                delta_loss,
                reward_loss,
                done_loss,
            ) = (
                self._train_dynamics_models()
            )

            # =============================================
            # GENERATE UR-MBPO SYNTHETIC DATA
            # =============================================

            accepted = (
                self._generate_synthetic_transitions()
            )

            attempted = min(
                self.synthetic_transitions,
                len(
                    self.real_buffer
                ),
            )

            rejected = (
                attempted
                -
                accepted
            )

            acceptance_rate = (
                100.0
                *
                accepted
                /
                max(
                    1,
                    attempted,
                )
            )

            # =============================================
            # LOGGING
            # =============================================

            print(
                "\n[UR-MBPO] "
                f"real_steps={timestep} | "
                f"real_buffer="
                f"{len(self.real_buffer)} | "
                f"attempted="
                f"{attempted} | "
                f"accepted="
                f"{accepted} | "
                f"uncertainty_rejected="
                f"{rejected} | "
                f"acceptance="
                f"{acceptance_rate:.1f}% | "
                f"synthetic_total="
                f"{self.total_synthetic}"
            )

            print(
                "[UR-MBPO] uncertainty | "
                f"min="
                f"{self.last_min_uncertainty:.6f} | "
                f"mean="
                f"{self.last_mean_uncertainty:.6f} | "
                f"max="
                f"{self.last_max_uncertainty:.6f} | "
                f"threshold="
                f"{self.uncertainty_threshold:.6f}"
            )

            print(
                "[UR-MBPO] risk | "
                "mean_predicted_clearance="
                f"{self.last_mean_predicted_clearance:.3f} m | "
                "min_predicted_clearance="
                f"{self.last_min_predicted_clearance:.3f} m | "
                "risk_penalized_total="
                f"{self.total_risk_penalized} | "
                "near_miss_predicted_total="
                f"{self.total_near_miss_predictions}"
            )

            print(
                "[UR-MBPO] totals | "
                f"candidates="
                f"{self.total_candidates} | "
                f"accepted="
                f"{self.total_accepted} | "
                f"uncertainty_rejected="
                f"{self.total_uncertainty_rejected}"
            )

            print(
                "[UR-MBPO] validation | "
                f"delta_mse="
                f"{delta_loss:.6f} | "
                f"reward_loss="
                f"{reward_loss:.6f} | "
                f"done_loss="
                f"{done_loss:.6f}"
            )

        return True

    # =====================================================
    # SAVE DYNAMICS
    # =====================================================

    def save_dynamics(
        self,
        path,
    ):
        """
        Save both the learned dynamics ensemble and all
        UR-MBPO configuration/statistics required for
        reproducibility.
        """

        checkpoint = {

            # =================================================
            # BASE MBPO CONFIGURATION
            # =================================================

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

            # =================================================
            # UR-MBPO STATISTICS
            # =================================================

            "total_candidates":
                self.total_candidates,

            "total_accepted":
                self.total_accepted,

            "total_uncertainty_rejected":
                self.total_uncertainty_rejected,

            "total_risk_penalized":
                self.total_risk_penalized,

            "total_near_miss_predictions":
                self.total_near_miss_predictions,

            # =================================================
            # UR-MBPO CONFIGURATION
            # =================================================

            "uncertainty_threshold":
                self.uncertainty_threshold,

            "uncertainty_penalty_scale":
                self.uncertainty_penalty_scale,

            "safe_clearance":
                self.safe_clearance,

            "near_miss_clearance":
                self.near_miss_clearance,

            "lidar_max_distance":
                self.lidar_max_distance,

            "risk_penalty_scale":
                self.risk_penalty_scale,

            "lidar_start_index":
                self.lidar_start_index,

            "num_lidar_sectors":
                self.num_lidar_sectors,

            # =================================================
            # DYNAMICS ENSEMBLE WEIGHTS
            # =================================================

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

        print(
            "\nUR-MBPO dynamics ensemble saved to:"
        )

        print(
            path
        )
