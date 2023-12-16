import copy

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import time
class Hamiltonian:
    def __init__(
        self,
        nparticles,
        dim,
        int_type,
        backend,
    ):
        """
        Note that this assumes that the wavefunction form is in the log domain
        """
        self._N = nparticles
        self._dim = dim
        self._int_type = int_type

        match backend:
            case "numpy":
                self.backend = np
                self.la = np.linalg
            case  "jax":
                self.backend = jnp
                self.la = jnp.linalg
                self.potential = jax.jit(self.potential)
            case _: # noqa
                raise ValueError("Invalid backend:", backend)

    # methods to be overwritten
    def potential(self, r):
        """Potential energy function"""
        raise NotImplementedError

    def _local_kinetic_energy(self, wf, r):
        """Evaluate the local kinetic energy of the system"""
        raise NotImplementedError

    def local_energy(self, wf, r):
        """Local energy of the system"""
        raise NotImplementedError

    def drift_force(self, wf, r):
        """Drift force at each particle's location"""
        raise NotImplementedError


class HarmonicOscillator(Hamiltonian):
    def __init__(
        self,
        nparticles,
        dim,
        int_type,
        backend,
        kwargs,
    ):
        """
        Note that nparticle and dim is part of the wavefunction.
        """
        super().__init__(nparticles, dim, int_type, backend)
        #self.potential = jax.jit(self.potential) # if we regularize we cannot jit

        self.kwargs = kwargs

    def potential(self, r):
        """Potential energy function"""
        # HO trap
        v_trap = 0.5 * self.backend.sum(r * r) * self.kwargs["omega"]

        # Interaction
        v_int = 0.0
        match self._int_type:
            case "Coulomb":

                r_cpy = copy.deepcopy(r).reshape(self._N, self._dim)
                r_dist = self.la.norm(r_cpy[None, ...] - r_cpy[:, None], axis=-1)

                v_int = self.backend.sum(
                    self.backend.triu(1 / r_dist, k=1)
                )  # k=1 to remove diagonal, since we don't want self-interaction
            case None:
                pass
            case _: # noqa
                raise ValueError("Invalid interaction type:", self._int_type)

        return v_trap + v_int


    def _local_kinetic_energy(self, wf, r):
        """Evaluate the local kinetic energy of the system"""
        _laplace = wf.laplacian(r).sum()  # summing over all particles
        _grad = wf.grad_wf(r)
        _grad2 = self.backend.sum(_grad * _grad)  # summing over all particles
        return -0.5 * (_laplace + _grad2)

    def local_energy(self, wf, r):
        """Local energy of the system"""

        ke = self._local_kinetic_energy(wf, r)
        pe = self.potential(r)

        return pe + ke

    def drift_force(self, wf, r):
        """Drift force at each particle's location"""
        F = 2 * wf.grad_wf(r)
        return F