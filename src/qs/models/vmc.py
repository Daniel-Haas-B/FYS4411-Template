import jax
import jax.numpy as jnp
import numpy as np
from qs.utils import Parameter # IMPORTANT: you may or may not use this depending on how you want to implement your code and especially your jax gradient implementation
from qs.utils import State


class VMC:
    def __init__(
        self,
        nparticles,
        dim,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="numpy",
    ):
        self._configure_backend(backend)
        self._initialize_vars(nparticles, dim, rng, log, logger, logger_level)

        r = 0 # initialize the positions randomly

        self._initialize_variational_params()
        self.state = 0 # take a look at the qs.utils State class

        if self.log:
            msg = f"""VMC initialized with {self._N} particles in {self._dim} dimensions with {
                    self.params.get("alpha").size
                    } parameters"""
            self._logger.info(msg)

    def _configure_backend(self, backend):
        """
        Here we configure the backend, for example, numpy or jax
        You can use this to change the linear alebra methods or do just in time compilation.
        Note however that depending on how you build your code, you might not need this.
        """
        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg

            # Here we overwrite the functions with their jax versions. This is just a suggestion.
            self.grad_wf_closure = self.grad_wf_closure_jax
            self.grads_closure = self.grads_closure_jax
            self.laplacian_closure = self.laplacian_closure_jax
            self._jit_functions()
        else:
            raise ValueError("Invalid backend:", backend)

    def _jit_functions(self):
        """
        Note there are other ways to jit functions. this is just one example.
        However, you should be careful with how you jit functions.
        They have to be pure functions, meaning they cannot have side effects (modify some state variable values outside its local environment)
        Take a close look at "https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html"
        """
        functions_to_jit = [
            "prob_closure",
            "wf",
            "grad_wf_closure",
            "laplacian_closure",
            "grads_closure",
        ]
        for func in functions_to_jit:
            setattr(self, func, jax.jit(getattr(self, func)))
        return self

    def wf(self, r, alpha):
        """
        
        r: (N, dim) array so that r_i is a dim-dimensional vector
        alpha: (N, dim) array so that alpha_i is a dim-dimensional vector

        return: should return Ψ(alpha, r)

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
        
        return 

    def prob_closure(self, r, alpha):
        """
        Return a function that computes |Ψ(alpha, r)|^2

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
        return 

    def prob(self, r):
        """
        Helper for the probability density

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
        alpha = 0 # get the parameters somehow
        return self.prob_closure(r, alpha)

    def grad_wf_closure(self, r, alpha):
        """
        Computes the gradient of the wavefunction with respect to r analytically
        """
        return 

    def grad_wf_closure_jax(self, r, alpha):
        """
        computes the gradient of the wavefunction with respect to r, but with jax grad
        """

        return

    def grad_wf(self, r):
        """
        Helper for the gradient of the wavefunction with respect to r

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
        alpha = 0 # get the parameters somehow

        return self.grad_wf_closure(r, alpha)

    def grads(self, r):
        """
        Helper for the gradient of the wavefunction with respect to the variational parameters

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
        alpha = 0 # get the parameters somehow

        return self.grads_closure(r, alpha)

    def grads_closure(self, r, alpha):
        """
        Computes the gradient of the wavefunction with respect to the variational parameters analytically
        """
        return 

    def grads_closure_jax(self, r, alpha):
        """
        Return a function that computes the gradient of the wavefunction with respect to the variational parameters with jax grad
        """

        return 

    def laplacian(self, r):
        """
        Return a function that computes the laplacian of the wavefunction ∇^2 Ψ(r)

        OBS: We strongly recommend you work with the wavefunction in log domain. 
        """
        alpha = 0 # get the parameters somehow
        return self.laplacian_closure(r, alpha)

    def laplacian_closure(self, r, alpha):
        """
        Analytical expression for the laplacian of the wavefunction
        """
        return 

    def laplacian_closure_jax(self, r, alpha):
        """
        Computes the laplacian of the wavefunction with jax grad
        """
        return

    def _initialize_vars(self, nparticles, dim, rng, log, logger, logger_level):
        self._N = nparticles
        self._dim = dim
        self._log = log if log else False

        if logger:
            self._logger = logger
        else:
            import logging
            self._logger = logging.getLogger(__name__)

        self._logger_level = logger_level
        self._rng = rng if rng else np.random.default_rng()

    def _initialize_variational_params(self, rng):
        # Initialize variational parameters in the correct range with the correct shape
        # Take a look at the qs.utils.Parameter class. You may or may not use it depending on how you implement your code.
        pass 