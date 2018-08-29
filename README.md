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

This will create all five executables mentioned aboved. If you're interested in only one of them, select the appropriate target e.g.
```sh
make conv\_it\_cuda
```

### Testing
To launch an automated test type
```sh
make test
```

You can change the settings of the test (such as version, number of threads/processes) in runAll.sh.

Copyright © 2018 Francesco Primon.
