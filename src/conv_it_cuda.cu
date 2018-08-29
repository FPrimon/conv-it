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

#include "parse.hpp"
#include "conv_it_def.hpp"
#include "conv_it_cuda_def.hpp"
#include "matrixConvCUDA.hpp"

const int ARGS = 8;

int main(int argc, char *argv[])
{
	if (argc != ARGS+1)
	{
		std::cerr << "Error: " << ARGS << " arguments needed" << std::endl;
		return EXIT_FAILURE;
	}

	const unsigned int iter = parse<unsigned int>(argv[1]);
	
	const dimType matRow = parse<dimType>(argv[2]);
	const dimType matCol = parse<dimType>(argv[3]);
	
	const dimType kerRow = parse<dimType>(argv[4]);
	const dimType kerCol = parse<dimType>(argv[5]);
	
	if (kerRow>matRow || kerCol>matCol)
	{
		std::cerr << "Error: invalid kernel dimension(s)" << std::endl;
		return EXIT_FAILURE;
	}
	if (kerRow*kerCol > kernelMaxNumberOfCells)
	{
		std::cerr << "Error: kernel is too large" << std::endl;
		std::cerr << "Please change \"kernelMaxNumberOfCells\" value in \"conv_it_cuda_def.hpp\" file" << std::endl;
		return EXIT_FAILURE;
	}
	
	MatrixCUDA mat(matRow,matCol,argv[6]);
	const MatrixCUDA ker(kerRow,kerCol,argv[7]);
	
	/*
	mat.kernelSetup();
	mat.check();
	return EXIT_FAILURE;
	*/
	
	if(mat.convolve(ker,iter) == EXIT_FAILURE)
	{
		std::cerr << "Error: unable to perform matrix convolution" << std::endl;
		return EXIT_FAILURE;
	}
	
	mat.toFile(argv[8]);
	
	return EXIT_SUCCESS;
}
