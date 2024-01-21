# import copy
import sys
import warnings

sys.path.insert(0, "../src/")

from qs.utils import errors
from qs.utils import generate_seed_sequence
from qs.utils import setup_logger
from qs.utils import State
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


import numpy as np
import pandas as pd

from qs.models import VMC

from numpy.random import default_rng
from tqdm.auto import tqdm

from physics.hamiltonians import HarmonicOscillator as HO

from samplers.metropolis import Metropolis as Metro

import optimizers as opt

warnings.filterwarnings("ignore", message="divide by zero encountered")


class QS:
    def __init__(
        self,
        backend="numpy",
        log=True,
        logger_level="INFO",
        rng=None,
        seed=None,
    ):
        """Quantum State
        It is conceptually important to understand that this is the system.
        The system is composed of a wave function, a hamiltonian, a sampler and an optimizer.
        This is the high level class that ties all the other classes together.
        """

        self._check_logger(log, logger_level)
        
        self._log = log
        self.hamiltonian = None
        self._backend = backend
        self.mcmc_alg = None
        self._optimizer = None
        self.wf = None
        self._seed = seed
        self.logger = setup_logger(self.__class__.__name__, level=logger_level) if self._log else None
        

        if rng is None:
            self.rng = default_rng

        # Suggestion of checking flags
        self._is_initialized_ = False
        self._is_trained_ = False
        self._sampling_performed = False

    def set_wf(self, wf_type, nparticles, dim, **kwargs):
        """
        Set the wave function to be used for sampling.
        For now we only support the VMC.
        Successfully setting the wave function will also initialize it 
        (this is because we expect the VMC class to initialize the variational parameters but you have to implement this yourself).
        """

        # check VMC script
        self._N = nparticles
        self._dim = dim
        self._wf_type = wf_type


        self._is_initialized_ = True

    def set_hamiltonian(self, type_, int_type, **kwargs):
        """
        Set the hamiltonian to be used for sampling.
        For now we only support the Harmonic Oscillator.

        Hamiltonian also needs to be propagated to the sampler if you at some point collect the local energy there.
        """

        # check HO script


    def set_sampler(self, mcmc_alg, scale=0.5):
        """
        Set the MCMC algorithm to be used for sampling.
        """
        self.mcmc_alg = mcmc_alg
        self._scale = scale

        # check metropolis sampler script


    def set_optimizer(self, optimizer, eta, **kwargs):
        """
        Set the optimizer algorithm to be used for param update.
        """
        self._eta = eta
        
        # check Gd script
        self._optimizer = opt.Gd(eta=eta)


    def train(self, max_iter, batch_size, **kwargs):
        """
        Train the wave function parameters.
        Here you should calculate sampler statistics and update the wave function parameters based on the derivative of the (statistical) local energy.
        """
        self._is_initialized()
        self._training_cycles = max_iter
        self._training_batch = batch_size

        if self._log:
            t_range = tqdm(
                range(max_iter),
                desc="[Training progress]",
                position=0,
                leave=True,
                colour="green",
            )
        else:
            t_range = range(max_iter)

        steps_before_optimize = batch_size

        epoch = 0
        for _ in t_range:
            # Here you collect batch_size samples and calculate the local energy
            # After you have collected batch_size samples, you update the parameters of the wave function
            
            
            steps_before_optimize -= 1
            if steps_before_optimize == 0:
                epoch += 1
            
            
                # Make Descent step with optimizer

            
                steps_before_optimize = batch_size

        
        self._is_trained_ = True
        if self.logger is not None:
            self.logger.info("Training done")


    def sample(self, nsamples, nchains=1, seed=None):
        """helper for the sample method from the Sampler class"""

        self._is_initialized() # check if the system is initialized
        self._is_trained() # check if the system is trained

        # Suggestion of things to display in the results
        system_info = {
            "nparticles": self._N,
            "dim": self._dim,
            "eta": self._eta,
            "mcmc_alg": self.mcmc_alg,
            "training_cycles": self._training_cycles,
            "training_batch": self._training_batch,
            "Opti": self._optimizer.__class__.__name__,
        }

        system_info = pd.DataFrame(system_info, index=[0])

        #OBS: this should actually be returned from the sampler sample method. This is as is below just a placeholder
        sample_results = {
            "chain_id": None,
            "energy": None,
            "std_error": None,
            "variance": None,
            "accept_rate": None,
            "scale": None,
            "nsamples": nsamples,
        }
        sample_results = pd.DataFrame(sample_results, index=[0])


        system_info_repeated = system_info.loc[
            system_info.index.repeat(len(sample_results))
        ].reset_index(drop=True)

        self._results = pd.concat([system_info_repeated, sample_results], axis=1)

        return self._results
    

    def _is_initialized(self):
        if not self._is_initialized_:
            msg = "A call to 'init' must be made before training"
            raise errors.NotInitialized(msg)

    def _is_trained(self):
        if not self._is_trained_:
            msg = "A call to 'train' must be made before sampling"
            raise errors.NotTrained(msg)

    def _sampling_performed(self):
        if not self._is_trained_:
            msg = "A call to 'sample' must be made in order to access results"
            raise errors.SamplingNotPerformed(msg)

    def _check_logger(self, log, logger_level):
        if not isinstance(log, bool):
            raise TypeError("'log' must be True or False")

        if not isinstance(logger_level, str):
            raise TypeError("'logger_level' must be passed as str")