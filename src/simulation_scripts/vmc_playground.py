import sys

sys.path.append("/Users/haas/Documents/Masters/TACompPhys/FYS4411-Template/src/") # append yout path to the src folder
import jax


from qs import quantum_state
import config


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")



# set up the system with its backend and level of logging, seed, and other general properties depending on how you want to run it
system = quantum_state.QS(
    backend=config.backend,
    log=True,
    logger_level="INFO",
    seed=config.seed,
)

# set up the wave function with some of its properties 
system.set_wf(
    config.wf_type,
    config.nparticles,
    config.dim,
)

# choose the sampler algorithm and scale
system.set_sampler(mcmc_alg=config.mcmc_alg, scale=1.0)

# choose the hamiltonian
system.set_hamiltonian(type_="ho", int_type="Coulomb", omega=1.0)

# choose the optimizer, learning rate, and other properties depending on the optimizer
system.set_optimizer(
    optimizer=config.optimizer,
    eta=config.eta,
)

# train the system, meaning we find the optimal variational parameters for the wave function
system.train(
    max_iter=config.training_cycles,
    batch_size=config.batch_size,
    seed=config.seed,
)

# now we get the results or do whatever we want with them
results = system.sample(config.nsamples, nchains=config.nchains, seed=config.seed)

# display the results
print(results)

