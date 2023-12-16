# import jax
import numpy as np
from qs.utils import advance_PRNG_state
from qs.utils import State

from .sampler import Sampler

# from jax import vmap


class Metropolis(Sampler):
    def __init__(self, rng, scale, logger=None):
        super().__init__(rng, scale, logger)

    def _step(self, wf, state, seed):
        """One step of the random walk Metropolis algorithm

        Parameters
        ----------
        state : qs.State
            Current state of the system. See state.py

        scale : float
            Scale of proposal distribution. Default: 0.5

        Returns
        -------
        new_state : qs.State
            The updated state of the system.

        # TODO: update the Metropolis Hastings and the fixed step and tunning
        """

        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Sample proposal positions, i.e., move walkers
        proposals = rng.normal(loc=state.positions, scale=self.scale)
        # print("self.scale", self.scale)
        # Sample log uniform rvs
        log_unif = np.log(rng.random())

        # Compute proposal log density

        logp_proposal = wf.logprob(proposals)

        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp

        # If accept is True, yield proposal, otherwise keep old state
        new_positions = proposals if accept else state.positions

        # Create new state
        new_logp = wf.logprob(new_positions) if accept else state.logp
        new_n_accepted = state.n_accepted + accept
        new_delta = state.delta + 1

        state.positions = new_positions
        state.logp = new_logp
        state.n_accepted = new_n_accepted
        state.delta = new_delta
        return state

    def _fixed_step(self, wf, state, seed, fixed_index=0):
        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Sample proposal positions, i.e., move walkers
        positions = state.positions
        proposals = rng.normal(loc=positions, scale=self.scale)
        proposals[fixed_index] = positions[fixed_index]  # Fix one particle
        # Sample log uniform rvs
        log_unif = np.log(rng.random())

        # Compute proposal log density
        logp_proposal = wf.logprob(proposals)

        # Metroplis acceptance criterion
        accept = log_unif < logp_proposal - state.logp

        # If accept is True, yield proposal, otherwise keep old state
        new_positions = proposals if accept else state.positions

        # Create new state
        new_logp = wf.logprob(new_positions)
        new_n_accepted = state.n_accepted + accept
        new_delta = state.delta + 1
        new_state = State(new_positions, new_logp, new_n_accepted, new_delta)

        return new_state

    def step(self, wf, state, seed):
        return self._step(wf, state, seed)

    # def batch_step(self, wf, state, seed):
    #     """Performs a Metropolis step on a batch of states.

    #     Parameters
    #     ----------
    #     batch_state : qs.State
    #         Current batch of states of the system.

    #     seed : int
    #         Seed for random number generator.

    #     Returns
    #     -------
    #     new_batch_state : qs.State
    #         The updated batch of states.
    #     """
    #     batch_size = 10
    #     batch_state = state.create_batch_of_states(batch_size)

    #     print("batch_seeds", batch_seeds)

    #     # Vectorize the _step function to apply it to each state in the batch
    #     vectorized_step = vmap(self._step, in_axes=(None, 0, None))

    #     # Apply the vectorized step function to the entire batch
    #     new_batch_state = vectorized_step(wf, batch_state, seed)
    #     print("new_batch_state", new_batch_state)
    #     exit()

    #     return new_batch_state

    def tune_scale(self, scale, acc_rate):
        """Proposal scale lookup table. (Original)

        Aims to obtain an acceptance rate between 20-50%.

        Retrieved from the source code of PyMC [1].

        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate over the last tune_interval:

                        Rate    Variance adaptation
                        ----    -------------------
                        <0.001        x 0.1
                        <0.05         x 0.5
                        <0.2          x 0.9
                        >0.5          x 1.1
                        >0.75         x 2
                        >0.95         x 10

        References
        ----------
        [1] https://github.com/pymc-devs/pymc/blob/main/pymc/step_methods/metropolis.py#L263

        Arguments
        ---------
        scale : float
            Scale of the proposal distribution
        acc_rate : float
            Acceptance rate of the last tuning interval

        Returns
        -------
        scale : float
            Updated scale parameter
        """
        # print("acc_rate: ", acc_rate)
        # print("scale before: ", scale)
        if acc_rate < 0.001:
            # reduce by 90 percent
            return scale * 0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            scale *= 0.5
        elif acc_rate < 0.2:
            # reduce by ten percent
            scale *= 0.9
        elif acc_rate > 0.5:
            # increase by ten percent
            scale *= 1.1
        elif acc_rate > 0.75:
            # increase by double
            scale *= 2.0
        elif acc_rate > 0.95:
            # increase by factor of ten
            scale *= 10.0
        # print("scale after: ", scale)
        self.scale = scale
        return scale