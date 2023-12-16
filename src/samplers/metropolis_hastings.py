import numpy as np
from qs.utils import advance_PRNG_state

from .sampler import Sampler


class MetroHastings(Sampler):
    def __init__(self, rng, scale, logger=None):
        super().__init__(rng, scale, logger)

    def _step(self, wf, state, seed):
        """One step of the Langevin Metropolis-Hastings algorithm

        Parameters
        ----------
        state : State
            Current state of the system. See state.py
        alpha :
            Variational parameter
        D : float
            Diffusion constant. Default: 0.5
        dt : float
            Scale of proposal distribution. Default: 1.0
        """

        # Precompute
        dt = self._scale**2
        Ddt = 0.5 * dt
        quarterDdt = 1 / (4 * Ddt)
        sys_size = state.positions.shape

        # Advance RNG
        next_gen = advance_PRNG_state(seed, state.delta)
        rng = self._rng(next_gen)

        # Compute drift force at current positions
        F = self.hamiltonian.drift_force(wf, state.positions)

        # Sample proposal positions, i.e., move walkers
        proposals = (
            state.positions
            + F * Ddt
            + rng.normal(loc=0, scale=self._scale, size=sys_size)
        )

        # Compute proposal log density
        logp_proposal = wf.logprob(proposals)

        # Green's function conditioned on proposals
        F_prop = self.hamiltonian.drift_force(wf, proposals)
        G_prop = -((state.positions - proposals - Ddt * F_prop) ** 2) * quarterDdt

        # Green's function conditioned on current positions
        G_cur = -((proposals - state.positions - Ddt * F) ** 2) * quarterDdt

        # Metroplis-Hastings ratio
        ratio = logp_proposal + np.sum(G_prop) - state.logp - np.sum(G_cur)

        # Sample log uniform rvs
        log_unif = np.log(rng.random())

        # Metroplis acceptance criterion
        accept = log_unif < ratio

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

    def step(self, wf, state, seed):
        return self._step(wf, state, seed)
