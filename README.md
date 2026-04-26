# NN Training with OpenMP/MPI

Sequential C++ baseline for the CS 5220 project. Trains a feedforward net on MNIST using Eigen. The plan is to layer MPI and OpenMP on top of this without changing `nn.h`.

## Dependencies

- C++17 compiler (clang++ on Mac, g++ on Linux/Perlmutter)
- Eigen 3.4

On Mac just `brew install eigen` and you're done. If you don't have brew or you're on Perlmutter, clone Eigen into the project root and the Makefile picks it up automatically:

```bash
git clone --branch 3.4.0 --depth 1 https://gitlab.com/libeigen/eigen.git
```

## Build and run

```bash
python3 download_mnist.py   # pulls MNIST into data/
make
./build/train
```
 
On Mac if `g++` isn't available: `make CXX=clang++`
 
### OpenMP (parallel) build
 
```bash
make OMP=1
./build/train_omp
```
 
To cap the number of threads (default is all logical cores):
 
```bash
make OMP=1 THREADS=8
./build/train_omp
```

Should hit ~97% test accuracy by epoch 10. Each epoch takes a few seconds on a laptop.

## Adding MPI

The net exposes `pack_gradients` / `unpack_gradients` so you can slot in an all-reduce without touching any of the forward/backward logic:

```cpp
net.backward(X_batch, Y_batch);

Eigen::VectorXd flat = net.pack_gradients();
MPI_Allreduce(MPI_IN_PLACE, flat.data(), flat.size(), MPI_DOUBLE, MPI_SUM, comm);
flat /= world_size;
net.unpack_gradients(flat);

optimizer.update(net, epoch);
```

To broadcast weights from rank 0 at startup:

```cpp
Eigen::VectorXd params = net.pack_params();
MPI_Bcast(params.data(), params.size(), MPI_DOUBLE, 0, comm);
net.unpack_params(params);
```

`net.total_params()` gives you the buffer length if you need it for MPI calls.

## Perlmutter

Eigen isn't in the default environment so clone it first (see above). Then:

```bash
module load PrgEnv-gnu
make CXX=g++
```

Run interactively:

```bash
salloc --nodes=1 --qos=interactive --time=00:30:00 --constraint=cpu -A <account>
./build/train
```

When you add MPI: `module load cray-mpich` and swap `CXX` for `mpicxx`. For OpenMP add `-fopenmp` to `CXXFLAGS`.
