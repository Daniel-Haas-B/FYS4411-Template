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
                # You might also be able to jit some functions here
            case _: # noqa
                raise ValueError("Invalid backend:", backend)

    def local_energy(self, wf, r):
        """Local energy of the system"""
        raise NotImplementedError



class HarmonicOscillator(Hamiltonian):
    def __init__(
        self,
        nparticles,
        dim,
        int_type,
        backend,
    ):
        
        
        super().__init__(nparticles, dim, int_type, backend)

    def local_energy(self, wf, r):
        """Local energy of the system
        
        Here you can use some methods from the wf class to compute the laplacian and gradients or do some other way you see fit.
        Note that receiving the wf object is just a suggestion - there might be other ways to do this.
        """

        ke = 0.0 # to be implemented
        pe = 0.0 # to be implemented

        return pe + ke