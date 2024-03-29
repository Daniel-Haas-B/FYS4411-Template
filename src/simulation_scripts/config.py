# Config
output_filename = "../data/vmc_playground.csv"

nparticles = 1
dim = 1
nsamples = int(2**18)  # 2**18 = 262144
nchains = 1 # number of Markov chains. When you parallelize, you can set this to the number of cores. Note you will have to implement this yourself.
eta = 0.01

training_cycles = 50_000  # this is cycles for the ansatz
mcmc_alg = "m"
backend = "numpy"
optimizer = "gd"
batch_size = 200
detailed = True
wf_type = "vmc"
seed = 142