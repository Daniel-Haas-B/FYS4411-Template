import sys
import time

import jax
import pandas as pd



sys.path.insert(0, "../qs/")
import qs  # noqa

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


output_filename = "../data/runtimes.csv"
nparticles = 2  # particles
dim = 2  # dimensionality
nhidden = 2  # hidden neurons
interaction = True

training_cycles = [10_000]  # , 20_000, 30_000, 40_000]
data = {
    "max_iter": training_cycles,
    "t_m_numpy": [],
    "t_lmh_numpy": [],
    "t_m_jax": [],
    "t_lmh_jax": [],
}

# Numpy, m
for max_iter in training_cycles:
    system = qs.RBM(
        nparticles,
        dim,
        nhidden=nhidden,
        interaction=interaction,
        mcmc_alg="m",
        qs_repr="psi",
        backend="numpy",
        log=False,
    )

    system.init(sigma2=1.0, scale=3.0)

    t_start = time.time()

    system.train(
        max_iter=max_iter,
        batch_size=1_000,
        optimizer="adam",
        eta=0.05,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    t_end = time.time()
    data["t_m_numpy"].append(t_end - t_start)


# Numpy, LMH
for max_iter in training_cycles:
    system = qs.RBM(
        nparticles,
        dim,
        nhidden=nhidden,
        interaction=interaction,
        mcmc_alg="lmh",
        qs_repr="psi",
        backend="numpy",
        log=False,
    )

    system.init(sigma2=1.0, scale=1.3)

    t_start = time.time()

    system.train(
        max_iter=max_iter,
        batch_size=1_000,
        optimizer="adam",
        eta=0.05,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    t_end = time.time()
    data["t_lmh_numpy"].append(t_end - t_start)


# JAX, m
for max_iter in training_cycles:
    system = qs.RBM(
        nparticles,
        dim,
        nhidden=nhidden,
        interaction=interaction,
        mcmc_alg="m",
        qs_repr="psi",
        backend="jax",
        log=False,
    )

    system.init(sigma2=1.0, scale=3.0)

    t_start = time.time()

    system.train(
        max_iter=max_iter,
        batch_size=1_000,
        optimizer="adam",
        eta=0.05,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    t_end = time.time()
    data["t_m_jax"].append(t_end - t_start)


# JAX, LMH
for max_iter in training_cycles:
    system = qs.RBM(
        nparticles,
        dim,
        nhidden=nhidden,
        interaction=interaction,
        mcmc_alg="lmh",
        qs_repr="psi",
        backend="jax",
        log=False,
    )

    system.init(sigma2=1.0, scale=1.3)

    t_start = time.time()

    system.train(
        max_iter=max_iter,
        batch_size=1_000,
        optimizer="adam",
        eta=0.05,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    t_end = time.time()
    data["t_lmh_jax"].append(t_end - t_start)

print(data)
df = pd.DataFrame(data)
print(df)
df.to_csv(output_filename, index=False)
# print(df[["energy", "std_error", "variance", "accept_rate"]])
