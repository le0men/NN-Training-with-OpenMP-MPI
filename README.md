# NN Training with OpenMP/MPI

A C++ feedforward neural network trained on MNIST that compares four all-reduce
algorithms for gradient aggregation in data-parallel SGD: hand-rolled tree,
ring, and halving-doubling implementations, plus MPI's own `MPI_Allreduce` as
a reference baseline. Also includes a non-blocking pipelined variant that
overlaps per-layer gradient communication with the backward pass.

Three binaries are produced:

- `build/train` — sequential / OpenMP baseline
- `build/train_mpi` — data-parallel SGD with selectable all-reduce algorithm
- `build/allreduce_bench` — standalone correctness + microbenchmark for the all-reduces
## Dependencies

- C++17 compiler (`g++` on Linux/Perlmutter, `clang++` on Mac)
- An MPI implementation (`mpicxx` on PATH). On Mac: `brew install open-mpi`.
  On Perlmutter: `module load PrgEnv-gnu cray-mpich`.
- Eigen 3.4. On Mac: `brew install eigen`. Otherwise clone it into the project root:

```bash
git clone --branch 3.4.0 --depth 1 https://gitlab.com/libeigen/eigen.git
```

## Build

```bash
python3 download_mnist.py        # pulls MNIST into data/
make all                         # builds train, train_mpi, allreduce_bench
```

Individual targets are available if you only need one of the binaries:

```bash
make seq                         # just the sequential / OpenMP target
make mpi                         # just the MPI target
make bench                       # just the all-reduce microbench
```

OpenMP is enabled by default in the MPI build so that each rank can use its own
threads. To produce a pure-MPI binary with OpenMP turned off:

```bash
make OMP=0 mpi
```

## Quickstart

```bash
# Sequential / OpenMP baseline
./build/train

# Data-parallel SGD with the ring all-reduce
mpirun -np 4 ./build/train_mpi --algo ring --epochs 10

# All-reduce correctness check + microbenchmark
mpirun -np 8 ./build/allreduce_bench --sizes 1024,65536,1048576
```

## `train_mpi` flags

```
--algo <tree|ring|hd|mpi>      all-reduce algorithm (default: ring)
--epochs <int>                 number of epochs (default: 10)
--global-batch <int>           total batch size across ranks (default: 64)
--lr <float>                   learning rate (default: 0.01)
--momentum <float>             SGD momentum (default: 0.9)
--layers <int,int,...>         network layer sizes, comma-separated
                               (default: 784,256,128,10)
--nonblocking                  pipeline per-layer iallreduce with backward
--verify                       per-iter verify hand-rolled all-reduce vs MPI_Allreduce
--warmup <int>                 skip first N iterations in timing (default: 0)
--train <path>                 train CSV (default: data/mnist_train.csv)
--test  <path>                 test  CSV (default: data/mnist_test.csv)
--verbose                      extra per-iter info on rank 0
-h, --help                     print usage and exit
```

## All-reduce algorithms

`include/allreduce.h` exposes four algorithms, all in-place sum reductions over
`MPI_DOUBLE`:

| algorithm           | latency steps | bytes/rank          | best regime                      |
|---------------------|---------------|---------------------|----------------------------------|
| `tree`              | 2 log P       | 2 N log P           | small messages                   |
| `ring`              | 2 (P − 1)     | 2 N (P − 1)/P       | large messages                   |
| `halving_doubling`  | 2 log P       | 2 N (P − 1)/P       | large messages, power-of-2 P     |
| `mpi`               | (impl)        | (impl)              | reference baseline               |

`tree` is binomial reduce-to-root + binomial broadcast. `ring` is the standard
chunk-by-chunk reduce-scatter / all-gather around a unidirectional ring.
`halving_doubling` is Rabenseifner: recursive-halving reduce-scatter followed
by recursive-doubling all-gather, falling back to `MPI_Allreduce` when P is not
a power of two.

The implementations live in `src/allreduce.cpp`. Each one takes a `MPI_Comm`
argument; `train_mpi` passes them a duplicate of `MPI_COMM_WORLD` so their
point-to-point traffic is isolated from the loss `MPI_Allreduce` and other MPI
calls in the iteration loop.

## `train_mpi` flags

```
--algo <tree|ring|hd|mpi>      all-reduce algorithm (default: ring)
--epochs <int>                 number of epochs (default: 10)
--global-batch <int>           total batch size across ranks (default: 64)
--lr <float>                   learning rate (default: 0.01)
--momentum <float>             SGD momentum (default: 0.9)
--nonblocking                  pipeline per-layer iallreduce with backward
--verify                       per-iter verify hand-rolled all-reduce vs MPI_Allreduce
--warmup <int>                 skip first N iterations in timing (default: 0)
--train <path>                 train CSV (default: data/mnist_train.csv)
--test  <path>                 test  CSV (default: data/mnist_test.csv)
```

## Benchmarking scripts

All scripts emit a CSV alongside their console output.

```bash
# Strong scaling for one algorithm: fix the global batch, vary rank count.
ALGO=ring RANKS="1 2 4 8" EPOCHS=3 ./scripts/strong_scaling.sh
#   -> scaling_ring.csv  (ranks, epoch, loss, acc, time_s, samples_per_s)

# All four algorithms at fixed P, plus the non-blocking pipelined variant.
P=8 EPOCHS=2 ./scripts/compare_algos.sh
#   -> compare_p8.csv    (algo, nonblocking, compute_ms, allreduce_ms, iter_ms, allreduce_pct)

# Vary network size (gradient buffer size) at a fixed rank count and algorithm.
P=8 ALGO=ring ./scripts/netsize_sweep.sh
#   -> netsize_p8_ring.csv (config, params_approx, algo, compute_ms, allreduce_ms, iter_ms, allreduce_pct)

# Alpha-beta sweep across message sizes (raw all-reduce, no training).
P=8 SIZES=1024,8192,65536,524288,4194304 REPS=30 ./scripts/microbench.sh
#   -> microbench_p8.csv (ranks, size, algo, median_ms, min_ms, gbs_eff)
```

## Perlmutter

Figures #3 and #4 on our report were made running one node:

```bash
salloc -N 1 -C cpu -q interactive -t 01:00:00 -A [account]
```
