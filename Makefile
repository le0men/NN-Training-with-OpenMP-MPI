CXX      ?= g++
CXXFLAGS := -O2 -std=c++17
TARGET   := build/train
SRC      := src/nn.cpp src/main.cpp

# check brew, system paths, then a local clone in the project root
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

ifeq ($(OMP),1)
  CXXFLAGS += -fopenmp
  LDFLAGS  += -fopenmp
  TARGET   := build/train_omp
  ifdef THREADS
    CXXFLAGS += -DOMP_NUM_THREADS_DEFAULT=$(THREADS)
    $(info OpenMP build enabled -> $(TARGET)  [threads: $(THREADS)])
  else
    $(info OpenMP build enabled -> $(TARGET)  [threads: runtime default])
  endif
else
  TARGET   := build/train
  $(info OpenMP build disabled -> $(TARGET)  [use OMP=1 to enable])
endif

.PHONY: all clean

all: build $(TARGET)

build:
	mkdir -p build

$(TARGET): $(SRC) include/nn.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRC)
	@echo "built $(TARGET)  [eigen: $(EIGEN_INC)]"

clean:
	rm -rf build
