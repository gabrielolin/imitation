"""AGGREVATE (https://arxiv.org/pdf/1406.5979).

Interactively trains policy by collecting some demonstrations, fitting a cost-to-go function, updating policy to maximize
cost to go. Initially the demonstrations just come from the
expert's policy; over time, they shift to be drawn more and more from the imitator's
policy.
"""

import abc
import logging
import os
import pathlib
import uuid
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from stable_baselines3.common import policies, utils, vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from torch.utils import data as th_data

from imitation.algorithms import base, bc
from imitation.data import rollout, serialize, types
from imitation.util import logger as imit_logger
from imitation.util import util
from imitation.algorithms.dagger import BetaSchedule, LinearBetaSchedule, InteractiveTrajectoryCollector, NeedsDemosException

class AggreVateTrainer(base.BaseImitationAlgorithm):
    _all_demos: List[types.Trajectory]

    DEFAULT_N_EPOCHS: int = 4
    """The default number of BC training epochs in `extend_and_update`."""

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        scratch_dir: types.AnyPath,
        rng: np.random.Generator,
        beta_schedule: Optional[Callable[[int], float]] = None,
        bc_trainer: bc.BC,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        """Builds AggreVateTrainer.

        Args:
            venv: Vectorized training environment.
            scratch_dir: Directory to use to store intermediate training
                information (e.g. for resuming training).
            rng: random state for random number generation.
            beta_schedule: Provides a value of `beta` (the probability of taking
                expert action in any given state) at each round of training. If
                `None`, then `linear_beta_schedule` will be used instead.
            bc_trainer: A `BC` instance used to train the underlying policy.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        super().__init__(custom_logger=custom_logger)

        if beta_schedule is None:
            beta_schedule = LinearBetaSchedule(15)
        self.beta_schedule = beta_schedule
        self.scratch_dir = util.parse_path(scratch_dir)
        self.venv = venv
        self.round_num = 0
        self._last_loaded_round = -1
        self._all_demos = []
        self.rng = rng

        utils.check_for_correct_spaces(
            self.venv,
            bc_trainer.observation_space,
            bc_trainer.action_space,
        )
        self.bc_trainer = bc_trainer
        self.bc_trainer.logger = self.logger

    def __getstate__(self):
        """Return state excluding non-pickleable objects."""
        d = dict(self.__dict__)
        del d["venv"]
        del d["_logger"]
        return d

    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        """Returns logger for this object."""
        return super().logger

    @logger.setter
    def logger(self, value: imit_logger.HierarchicalLogger) -> None:
        # DAgger and inner-BC logger should stay in sync
        self._logger = value
        self.bc_trainer.logger = value

    @property
    def policy(self) -> policies.BasePolicy:
        return self.bc_trainer.policy

    @property
    def batch_size(self) -> int:
        return self.bc_trainer.batch_size

    def _load_all_demos(self) -> Tuple[types.Transitions, List[int]]:
        num_demos_by_round = []
        for round_num in range(self._last_loaded_round + 1, self.round_num + 1):
            round_dir = self._demo_dir_path_for_round(round_num)
            demo_paths = self._get_demo_paths(round_dir)
            self._all_demos.extend(serialize.load(p)[0] for p in demo_paths)
            num_demos_by_round.append(len(demo_paths))
        logging.info(f"Loaded {len(self._all_demos)} total")
        demo_transitions = rollout.flatten_trajectories(self._all_demos)
        return demo_transitions, num_demos_by_round

    def _get_demo_paths(self, round_dir: pathlib.Path) -> List[pathlib.Path]:
        # listdir returns filenames in an arbitrary order that depends on the
        # file system implementation:
        # https://stackoverflow.com/questions/31534583/is-os-listdir-deterministic
        # To ensure the order is consistent across file systems,
        # we sort by the filename.
        filenames = sorted(os.listdir(round_dir))
        return [round_dir / f for f in filenames if f.endswith(".npz")]

    def _demo_dir_path_for_round(self, round_num: Optional[int] = None) -> pathlib.Path:
        if round_num is None:
            round_num = self.round_num
        return self.scratch_dir / "demos" / f"round-{round_num:03d}"

    def _try_load_demos(self) -> None:
        """Load the dataset for this round into self.bc_trainer as a DataLoader."""
        demo_dir = self._demo_dir_path_for_round()
        demo_paths = self._get_demo_paths(demo_dir) if demo_dir.is_dir() else []
        if len(demo_paths) == 0:
            raise NeedsDemosException(
                f"No demos found for round {self.round_num} in dir '{demo_dir}'. "
                f"Maybe you need to collect some demos? See "
                f".create_trajectory_collector()",
            )

        if self._last_loaded_round < self.round_num:
            transitions, num_demos = self._load_all_demos()
            logging.info(
                f"Loaded {sum(num_demos)} new demos from {len(num_demos)} rounds",
            )
            if len(transitions) < self.batch_size:
                raise ValueError(
                    "Not enough transitions to form a single batch: "
                    f"self.batch_size={self.batch_size} > "
                    f"len(transitions)={len(transitions)}",
                )
            data_loader = th_data.DataLoader(
                transitions,
                self.batch_size,
                drop_last=True,
                shuffle=True,
                collate_fn=types.transitions_collate_fn,
            )
            self.bc_trainer.set_demonstrations(data_loader)
            self._last_loaded_round = self.round_num

    def extend_and_update(
        self,
        bc_train_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Extend internal batch of data and train BC.

        Specifically, this method will load new transitions (if necessary), train
        the model for a while, and advance the round counter. If there are no fresh
        demonstrations in the demonstration directory for the current round, then
        this will raise a `NeedsDemosException` instead of training or advancing
        the round counter. In that case, the user should call
        `.create_trajectory_collector()` and use the returned
        `InteractiveTrajectoryCollector` to produce a new set of demonstrations for
        the current interaction round.

        Arguments:
            bc_train_kwargs: Keyword arguments for calling `BC.train()`. If
                the `log_rollouts_venv` key is not provided, then it is set to
                `self.venv` by default. If neither of the `n_epochs` and `n_batches`
                keys are provided, then `n_epochs` is set to `self.DEFAULT_N_EPOCHS`.

        Returns:
            New round number after advancing the round counter.
        """
        if bc_train_kwargs is None:
            bc_train_kwargs = {}
        else:
            bc_train_kwargs = dict(bc_train_kwargs)

        user_keys = bc_train_kwargs.keys()
        if "log_rollouts_venv" not in user_keys:
            bc_train_kwargs["log_rollouts_venv"] = self.venv

        if "n_epochs" not in user_keys and "n_batches" not in user_keys:
            bc_train_kwargs["n_epochs"] = self.DEFAULT_N_EPOCHS

        logging.info("Loading demonstrations")
        self._try_load_demos()
        logging.info(f"Training at round {self.round_num}")
        self.bc_trainer.train(**bc_train_kwargs)
        self.round_num += 1
        logging.info(f"New round number is {self.round_num}")
        return self.round_num

    def create_trajectory_collector(self) -> InteractiveTrajectoryCollector:
        """Create trajectory collector to extend current round's demonstration set.

        Returns:
            A collector configured with the appropriate beta, imitator policy, etc.
            for the current round. Refer to the documentation for
            `InteractiveTrajectoryCollector` to see how to use this.
        """
        save_dir = self._demo_dir_path_for_round()
        beta = self.beta_schedule(self.round_num)
        collector = InteractiveTrajectoryCollector(
            venv=self.venv,
            get_robot_acts=lambda acts: self.bc_trainer.policy.predict(acts)[0],
            beta=beta,
            save_dir=save_dir,
            rng=self.rng,
        )
        return collector

    def save_trainer(self) -> Tuple[pathlib.Path, pathlib.Path]:
        """Create a snapshot of trainer in the scratch/working directory.

        The created snapshot can be reloaded with `reconstruct_trainer()`.
        In addition to saving one copy of the policy in the trainer snapshot, this
        method saves a second copy of the policy in its own file. Having a second copy
        of the policy is convenient because it can be loaded on its own and passed to
        evaluation routines for other algorithms.

        Returns:
            checkpoint_path: a path to one of the created `DAggerTrainer` checkpoints.
            policy_path: a path to one of the created `DAggerTrainer` policies.
        """
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        # save full trainer checkpoints
        checkpoint_paths = [
            self.scratch_dir / f"checkpoint-{self.round_num:03d}.pt",
            self.scratch_dir / "checkpoint-latest.pt",
        ]
        for checkpoint_path in checkpoint_paths:
            torch.save(self, checkpoint_path)

        # save policies separately for convenience
        policy_paths = [
            self.scratch_dir / f"policy-{self.round_num:03d}.pt",
            self.scratch_dir / "policy-latest.pt",
        ]
        for policy_path in policy_paths:
            util.save_policy(self.policy, policy_path)

        return checkpoint_paths[0], policy_paths[0]
