# import jax
import numpy as np
from qs.utils import advance_PRNG_state
from qs.utils import State

from .sampler import Sampler


class Metropolis(Sampler):
    def __init__(self, rng, scale, logger=None):
        super().__init__(rng, scale, logger)

    def _step(self, wf, state, seed):
        """One step of the random walk Metropolis algorithm

        *Suggested* parameters
        ----------
        wf : qs.WaveFunction object to get probability density from

        state : qs.State
            Current state of the system. See state.py

        scale : float
            Scale of proposal distribution.

        Returns
        -------
        new_state : qs.State
            The updated state of the system.

        """

        # You might want to Advance RNG as in 
        # next_gen = advance_PRNG_state(seed, state.delta)
        # rng = self._rng(next_gen)

        new_state = State() # this is just a placeholder
        return new_state


    def step(self, wf, state, seed):
        return self._step(wf, state, seed)
