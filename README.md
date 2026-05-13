# NN Training with OpenMP/MPI

C++ feedforward net trained on MNIST, with three back-ends:

- `build/train`            — sequential / OpenMP baseline
- `build/train_mpi`        — data-parallel SGD with selectable all-reduce algorithm
- `build/allreduce_bench`  — standalone correctness + microbenchmark for the all-reduces

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
make seq                         # just the sequential / OpenMP target
make mpi                         # just the MPI target
make bench                       # just the all-reduce microbench
```

To disable OpenMP within MPI ranks (pure MPI):

```bash
make OMP=0 mpi
```

## Run

```bash
./build/train                                                    # sequential
mpirun -np 4 ./build/train_mpi --algo ring --epochs 10           # data-parallel SGD
mpirun -np 8 ./build/allreduce_bench --sizes 1024,65536,1048576  # microbench
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

`train_mpi` shards the global batch across ranks: with `--global-batch 64` and
`-np 4`, each rank processes 16 samples per step. After backward, gradients are
summed across ranks via the chosen algorithm and divided by world_size, so the
parameter update is mathematically identical to single-rank training on the
full batch.

`--nonblocking` swaps in a custom backward (`backward_pipelined` in
`main_mpi.cpp`) that fires `MPI_Iallreduce` on layer L's gradients as soon as
they're computed, then walks down to layer L−1. All requests are awaited just
before the optimizer step. This overlaps the all-reduce of later layers with
the gradient computation of earlier ones.

`--verify` recomputes each iteration's all-reduce with `MPI_Allreduce` on a
copy and aborts if the results differ. Slow; use it once after any change to
the algorithms.

At the end of a run, `train_mpi` prints a per-iteration breakdown:

```
--- timing summary (95 iters, warmup 5 skipped) ---
avg compute / iter:    16.479 ms
avg all-reduce/iter:   46.059 ms  (70.2% of iter)
avg total / iter:      65.649 ms
gradient buffer:      235146 doubles (1.88 MB)
```

## Benchmarking scripts

All scripts emit a CSV alongside their console output.

```bash
# strong scaling for one algorithm
ALGO=ring RANKS="1 2 4 8" EPOCHS=3 ./scripts/strong_scaling.sh
#   -> scaling_ring.csv  (ranks, epoch, loss, acc, time_s, samples_per_s)

# all four algorithms at fixed P, plus the non-blocking pipelined variant
P=8 EPOCHS=2 ./scripts/compare_algos.sh
#   -> compare_p8.csv    (algo, nonblocking, compute_ms, allreduce_ms, iter_ms, allreduce_pct)

# alpha-beta sweep across message sizes
P=8 SIZES=1024,8192,65536,524288,4194304 REPS=30 ./scripts/microbench.sh
#   -> microbench_p8.csv (ranks, size, algo, median_ms, min_ms, gbs_eff)
```

## Adding MPI elsewhere

```cpp
#include "allreduce.h"

MPI_Comm reduce_comm = allreduce::dup_comm(MPI_COMM_WORLD);
// ...

net.backward(X_batch, Y_batch);

Eigen::VectorXd flat = net.pack_gradients();
allreduce::run(allreduce::Algorithm::RING, flat.data(), flat.size(), reduce_comm);
flat /= world_size;
net.unpack_gradients(flat);

optimizer.update(net, epoch);

// before MPI_Finalize:
MPI_Comm_free(&reduce_comm);
```

To broadcast weights from rank 0 at startup:

```cpp
Eigen::VectorXd params = net.pack_params();
MPI_Bcast(params.data(), params.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
net.unpack_params(params);
```

## Perlmutter

```bash
module load PrgEnv-gnu cray-mpich
git clone --branch 3.4.0 --depth 1 https://gitlab.com/libeigen/eigen.git
make all CXX=g++ MPICXX=CC

salloc --nodes=2 --ntasks-per-node=4 --qos=interactive --time=00:30:00 \
       --constraint=cpu -A <account>
srun -n 8 ./build/train_mpi --algo ring --epochs 10 --global-batch 256
```

For hybrid MPI+OpenMP runs, `OMP_NUM_THREADS` controls the thread count per
rank; pin ranks with `srun --cpu-bind=cores`.
