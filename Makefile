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

.PHONY: all clean

all: build $(TARGET)

build:
	mkdir -p build

$(TARGET): $(SRC) include/nn.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRC)
	@echo "built $(TARGET)  [eigen: $(EIGEN_INC)]"

clean:
	rm -rf build
