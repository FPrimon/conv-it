# conv-it

## Introduction

[*conv-it*](https://github.com/FPrimon/conv-it "GitHub repository") is a program for computing iterative convolution of integer matrices.

Originally it was an assignment for the *Algorithms and Parallel Computing* course I attended at [Politecnico di Milano](https://www.polimi.it/en).
Its purpose was to apply different parallelization methods to the same program. That's why there are five different versions and their respective executable files:

1. *conv\_it\_st*: single-threaded program
2. *conv\_it\_omp*: parallelism is achieved via [OpenMP](https://www.openmp.org/ "Open Multi-Processing")
3. *conv\_it\_mpi*: [MPI](https://www.mpi-forum.org/ "Message Parsing Interface") is used to distribute workload among processors
4. *conv\_it\_hybrid*: enhanced by both OpenMP and MPI
5. *conv\_it\_cuda*: runs on [CUDA](https://developer.nvidia.com/cuda-zone) capable GPU

## Instructions

### Before compiling
Make sure you've installed g++(supporting -std=c++11), make, OpenMP, MPI and CUDA (where available).

### Compiling
Once you have downloaded a copy of the repository run
```sh
make compile
```

This will create all five executables mentioned above. If you're interested in only one of them, select the appropriate target e.g.
```sh
make conv_it_cuda
```

### Testing
To launch an automated test, type
```sh
make test
```

You can change the settings of the test (such as version, number of threads/processes) by modifying directly the file *runAll.sh*.

### Passing arguments
In order to run properly, every *conv_it* executable needs the following arguments:

1. no. of iterations
2. no. of rows in the input matrix
3. no. of columns in the input matrix
4. no. of rows in the kernel
5. no. of columns in the kernel
6. name of the file containing the input matrix
7. name of the file containing the kernel
8. name of the output file

All files must contain *space-separated* integer values representing the entries of the matrix.

OpenMP needs you to set the OMP\_NUM\_THREADS variable.
MPI is enabled by launching the executable with
```sh
mpirun -np P executable argument1 argument2 ...
```
where P is the number of processes.

---

Tested on Ubuntu 16.04 LTS with gcc 5.4.0, GNU Make 4.1, OpenMPI 1.10.2, CUDA 9.2
Copyright © 2018 Francesco Primon.
