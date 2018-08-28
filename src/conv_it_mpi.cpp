/*
 *  This file is part of 'Conv-It' (https://github.com/FPrimon/conv-it).
 *  'Conv-It' is a program for iterative matrix convolution.
 *  Copyright © 2018 Francesco Primon.
 *
 *  'Conv-It' is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 *  'Conv-It' is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with 'Conv-It'. If not, see <https://www.gnu.org/licenses/>.
 */
#include <cstdlib>
#include <iostream>
#include <mpi.h>

#include "parse.hpp"
#include "stopwatch.hpp"
#include "conv_it_def.hpp"
#include "matrixConvMPI.hpp"

constexpr int ARGS = 8;

int main(int argc, char *argv[])
{
	if (argc != ARGS+1)
	{
		std::cerr << "Error: " << ARGS << " arguments needed" << std::endl;
		return EXIT_FAILURE;
	}
	
	MPIinitializer(&argc,&argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	stopwatch swTot;
	if (rank == 0)
	{
		swTot.reset();
		swTot.start();
	}
	
	const unsigned int iter = parse<unsigned int>(argv[1]);
	
	const dimType matRow = parse<dimType>(argv[2]);
	const dimType matCol = parse<dimType>(argv[3]);
	
	const dimType kerRow = parse<dimType>(argv[4]);
	const dimType kerCol = parse<dimType>(argv[5]);
	
	if (kerRow%2==0 || kerCol%2==0)
	{
		if (rank == 0)
			std::cerr << "Error: kernel dimensions must be odd values" << std::endl;
		
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	if (kerRow>matRow || kerCol>matCol)
	{
		if (rank == 0)
			std::cerr << "Error: invalid kernel dimension(s)" << std::endl;
		
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	
	Matrix ker(kerRow,kerCol);
	MatrixMPI mat(matRow,matCol,ker);
	stopwatch sw;
	
	if (rank == 0)
	{
		mat.fromFile(matRow,matCol,argv[6]);
		ker.fromFile(kerRow,kerCol,argv[7]);
		
		sw.reset();
		sw.start();
	}
	
	/*for(int r=0; r<18; ++r)
	{
		if (rank==r)
			mat.check();	
		MPI_Barrier(MPI_COMM_WORLD);
	}*/
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(mat.convolve(ker,iter) == EXIT_FAILURE)
	{
		if(rank == 0)
			std::cerr << "Error: unable to perform matrix convolution" << std::endl;
		
		MPI_Finalize();
		return EXIT_FAILURE;
	}
	
	if (rank == 0)
	{
		sw.stop();
		std::clog << "Elapsed time: " << sw.elapsed_ms() << " ms" << std::endl;
		
		mat.toFile(argv[8]);
		
		swTot.stop();
		std::clog << "Total elapsed time: " << swTot.elapsed_ms() << " ms" << std::endl;
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
