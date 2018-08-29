##
 #  This file is part of 'Conv-It' (https://github.com/FPrimon/conv-it).
 #  'Conv-It' is a program for iterative matrix convolution.
 #  Copyright © 2018 Francesco Primon.
 #
 #  'Conv-It' is free software: you can redistribute it and/or modify
 #  it under the terms of the GNU General Public License as published by
 #  the Free Software Foundation, either version 3 of the License, or
 #  any later version.
 #
 #  'Conv-It' is distributed in the hope that it will be useful,
 #  but WITHOUT ANY WARRANTY; without even the implied warranty of
 #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 #  GNU General Public License for more details.
 #
 #  You should have received a copy of the GNU General Public License
 #  along with 'Conv-It'. If not, see <https://www.gnu.org/licenses/>.
 ##
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

####### COMPILER SETTINGS: DO NOT CHANGE!!! ####### 

CXX = g++
MPICXX = mpic++
NVXX = nvcc -ccbin g++

CXXFLAGS = -O2 -march=native -Wall -Wextra -I$(INCLUDE_DIR) -I$(CUDA_DIR) -fopenmp
NVXXFLAGS = -m${OS_SIZE} -I$(INCLUDE_DIR) -I$(CUDA_DIR)
NVXXFLAGS += $(addprefix -Xcompiler ,$(CXXFLAGS))

CXX11FLAGS = -std=c++11
#NVXX11FLAGS = -Xcompiler -std=c++11	#non supportato?

#LIBRARIES = /usr/local/cuda/lib64
#LNKFLAGS = -lcudadevrt -L$(LIBRARIES) -lcudart

# CUDA code generation flags
#GENCODE_SM10    := -gencode arch=compute_10,code=sm_10	#deprecated
#GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM32    := -gencode arch=compute_32,code=sm_32
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SMXX    := -gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?= $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX)

#GENCODE_FLAGS = -arch=sm_20

#####################################################


############### HEADER FILE SETTINGS ###############

# PUT HERE THE FOLDER WITH YOUR HEADER FILES, IF ANY
# FOR EXAMPLE:
# INCLUDE_DIR = include
INCLUDE_DIR = include
CUDA_DIR = /usr/local/cuda/include

# PUT HERE THE PATH TO YOUR HEADER FILES TO CORRECTLY RECOMPILE AT ANY CHANGE, IF ANY
# FOR EXAMPLE:
# DEPS = $(INCLUDE_DIR)/*.h*
DEPS = $(INCLUDE_DIR)/conv_it_def.hpp $(INCLUDE_DIR)/matrixConv.hpp $(INCLUDE_DIR)/parse.hpp $(INCLUDE_DIR)/stopwatch.hpp
OMP_DEPS = $(INCLUDE_DIR)/matrixConvOMP.hpp
MPI_DEPS = $(INCLUDE_DIR)/matrixConvMPI.hpp
HYBRID_DEPS = $(INCLUDE_DIR)/matrixConvHybrid.hpp
CUDA_DEPS = $(INCLUDE_DIR)/conv_it_def.hpp $(INCLUDE_DIR)/conv_it_cuda_def.hpp $(INCLUDE_DIR)/matrixConvCUDA.hpp $(INCLUDE_DIR)/parse.hpp

#####################################################

############## SOURCE FILE SETTINGS ###############

# PUT HERE THE FOLDER WITH YOUR SOURCe FILES
# FOR EXAMPLE:
# SRC = src
SRC = src

# FOR EVERY VERSION OF THE PROGRAM, PUT HERE THE LIST OF SOURCE FILES TO BE COMPILED
# FOR EXAMPLE
# SERIAL_SRC = $(SRC)/conv_it_st.cpp $(SRC)/main_st.cpp

SERIAL_SRC = $(SRC)/conv_it_st.cpp

OMP_SRC = $(SRC)/conv_it_omp.cpp

MPI_SRC = $(SRC)/conv_it_mpi.cpp

HYBRID_SRC = $(SRC)/conv_it_hybrid.cpp

#CUDA_CPP_SRC = $(SRC)/conv_it_cuda.cpp
#CUDA_CU_SRC = $(SRC)/conv_it_cuda_def.cu

CUDA_SRC = $(SRC)/conv_it_cuda.cu

#####################################################

########### MAKE RULES: DO NOT CHANGE!!! ############

all: compile

compile: conv_it_st conv_it_omp conv_it_mpi conv_it_hybrid conv_it_cuda

conv_it_st: $(SERIAL_SRC) $(DEPS)
	$(CXX) $(CXXFLAGS) $(CXX11FLAGS) $(SERIAL_SRC) -o $@
	
conv_it_omp: $(OMP_SRC) $(DEPS) $(OMP_DEPS)
	$(CXX) $(CXXFLAGS) $(CXX11FLAGS) $(OMP_SRC) -o $@
	
conv_it_mpi: $(MPI_SRC) $(DEPS) $(MPI_DEPS)
	$(MPICXX) $(CXXFLAGS) $(CXX11FLAGS) $(MPI_SRC) -o $@
	
conv_it_hybrid: $(HYBRID_SRC) $(DEPS) $(HYBRID_DEPS)
	$(MPICXX) $(CXXFLAGS) $(CXX11FLAGS) $(HYBRID_SRC) -o $@

#conv_it_cuda_cu.o: $(CUDA_CU_SRC) $(DEPS)
#	$(NVXX) $(NVXXFLAGS) $(GENCODE_FLAGS) -dc $(CUDA_CU_SRC) -o $@
#	$(NVXX) $(NVXXFLAGS) $(GENCODE_FLAGS) -dlink $@ -lcudadevrt -o link.o

#conv_it_cuda_cpp.o: $(CUDA_CPP_SRC) $(DEPS)
#	$(CXX) $(CXXFLAGS) $(CXX11FLAGS) $(CUDA_CPP_SRC) -o $@ -c

conv_it_cuda: $(CUDA_SRC) $(CUDA_DEPS)
	$(NVXX) $(NVXXFLAGS) $(NVXX11FLAGS) $(GENCODE_FLAGS) $(CUDA_SRC) -o $@

test:
	bash runAll.sh

clean:
	rm -rf conv_it_* log/ data/output/cuda* data/output/hybrid* data/output/st* data/output/omp* data/output/mpi*
	
#####################################################
