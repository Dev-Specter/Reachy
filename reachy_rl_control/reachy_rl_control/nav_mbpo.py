#!/usr/bin/env python3
"""
MBPO callback for the V3 navigation environment.

Identical to mbpo_dynamics.MBPOCallback except for ONE line: the learned
model's terminal target now includes collisions.

    frozen mbpo_dynamics.py :  model_done = reason in ("goal", "fall")
    this file               :  model_done = reason in ("goal", "collision", "fall")

In V3 a collision ends the real episode, so the dynamics model must learn
that it is terminal; otherwise synthetic rollouts continue "through"
obstacles and SAC learns from impossible transitions. Timeouts remain
non-terminal (artificial horizon). The frozen file is left unchanged so
earlier results stay reproducible.
"""

import numpy as np

from reachy_rl_control.mbpo_dynamics import MBPOCallback

TERMINAL_REASONS = ("goal", "collision", "fall")


class NavMBPOCallback(MBPOCallback):

    def _store_real_transition(self):
        actions = np.asarray(self.locals["actions"], dtype=np.float32)
        new_obs = np.asarray(self.locals["new_obs"], dtype=np.float32)
        rewards = np.asarray(self.locals["rewards"], dtype=np.float32)
        dones = np.asarray(self.locals["dones"])
        info = self.locals["infos"][0]

        if bool(dones[0]) and "terminal_observation" in info:
            next_obs = np.asarray(info["terminal_observation"], dtype=np.float32).copy()
        else:
            next_obs = new_obs[0].copy()

        model_done = info.get("termination_reason", "") in TERMINAL_REASONS

        self.real_buffer.append((
            self.previous_obs.copy(),
            actions[0].copy(),
            float(rewards[0]),
            next_obs,
            float(model_done),
        ))

        # After an automatic VecEnv reset, new_obs already holds the reset observation.
        self.previous_obs = new_obs[0].copy()
