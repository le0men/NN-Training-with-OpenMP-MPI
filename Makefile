# ----- toolchain ------------------------------------------------------------
CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS := -O3 -std=c++17 -Wall -Wno-unused-but-set-variable

# Add OpenMP for the MPI build by default. Override with `make OMP=0` to disable.
OMP      ?= 1
ifneq ($(OMP),0)
  OMP_FLAG := -fopenmp
else
  OMP_FLAG :=
endif

# ----- locate Eigen ---------------------------------------------------------
# Check brew, system paths, then a local clone in the project root.
EIGEN_CANDIDATES := \
    $(shell brew --prefix eigen 2>/dev/null)/include/eigen3 \
    /usr/local/include/eigen3 \
    /usr/include/eigen3 \
    eigen

EIGEN_INC := $(firstword $(foreach dir,$(EIGEN_CANDIDATES),\
    $(if $(wildcard $(dir)/Eigen/Dense),$(dir),)))

ifndef EIGEN_INC
$(error Eigen not found. Run: git clone --branch 3.4.0 --depth 1 https://gitlab.com/libeigen/eigen.git)
endif

INCLUDES := -I include -I $(EIGEN_INC)

# ----- targets --------------------------------------------------------------
SEQ_TARGET := build/train
MPI_TARGET := build/train_mpi
ARB_TARGET := build/allreduce_bench

NN_SRC      := src/nn.cpp
SEQ_SRC     := $(NN_SRC) src/main.cpp
MPI_SRC     := $(NN_SRC) src/main_mpi.cpp src/allreduce.cpp
ARB_SRC     := src/allreduce_bench.cpp src/allreduce.cpp

.PHONY: all clean seq mpi bench

all: seq mpi bench

seq:   build $(SEQ_TARGET)
mpi:   build $(MPI_TARGET)
bench: build $(ARB_TARGET)

build:
	@mkdir -p build

# Sequential / OpenMP-only build (drop-in replacement for the original target).
$(SEQ_TARGET): $(SEQ_SRC) include/nn.h
	$(CXX) $(CXXFLAGS) $(OMP_FLAG) $(INCLUDES) -o $@ $(SEQ_SRC) $(OMP_FLAG)
	@echo "built $@  [eigen: $(EIGEN_INC), openmp: $(OMP)]"

# MPI build with optional OpenMP within each rank.
$(MPI_TARGET): $(MPI_SRC) include/nn.h include/allreduce.h
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAG) $(INCLUDES) -o $@ $(MPI_SRC) $(OMP_FLAG)
	@echo "built $@  [eigen: $(EIGEN_INC), openmp: $(OMP)]"

# Standalone all-reduce correctness + microbenchmark.
$(ARB_TARGET): $(ARB_SRC) include/allreduce.h
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(ARB_SRC)
	@echo "built $@"

clean:
	rm -rf build
