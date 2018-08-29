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
#ifndef MATRIX_CONV_HYBRID
#define MATRIX_CONV_HYBRID

#include <cstdlib>
#include <iostream>

#include "matrixConvOMP.hpp"
#include "matrixConvMPI.hpp"

//	CLASSes DECLARATIONs
class MatrixHybrid;

//	CLASSes DEFINITIONs
class MatrixHybrid : public MatrixOMP, public MatrixMPI	// Matrix is a virtual base
{
	//	CONSTRUCTORs
	public:
		MatrixHybrid() = default;		// costruisce la matrice 0x0
		MatrixHybrid(const dimType nR, const dimType nC, const ChunkMPI p) : Matrix(nR,nC), MatrixOMP(nR,nC), MatrixMPI(nR,nC,p)	{}
		MatrixHybrid( const dimType nR, const dimType nC, const Matrix ker) : Matrix(nR,nC), MatrixOMP(nR,nC), MatrixMPI(nR,nC,ker)	{}
	
	//	COPY-CONSTRUCTOR
	public:
		MatrixHybrid(const MatrixHybrid &mat) : Matrix(mat), MatrixOMP(mat), MatrixMPI(mat)	{}
		
	//	OPERATORs
	public:
		MatrixHybrid& operator=(const MatrixHybrid &rightMat)
		{
			Matrix::operator=(rightMat);
			MatrixOMP::operator=(rightMat);
			MatrixMPI::operator=(rightMat);
			return *this;
		}
	
	//	MEMBER-FUNCTIONs DECLARATIONs
	protected:
		int convolve(const Matrix ker)	override;
	public:
		int convolve(const Matrix ker, const unsigned int iter) override;
	
	//	DESTRUCTOR
	public:
		~MatrixHybrid() = default;
};

//	MEMBER-FUNCTIONs DEFINITIONs
inline int MatrixHybrid::convolve(const Matrix ker)
{
	if (reset() == nullptr)
		std::cerr << "Error in MatrixHybrid::convolve call: matrix does not exist" << std::endl;
	
	else if (ker.reset() == nullptr)
		std::cerr << "Error in MatrixHybrid::convolve call: kernel does not exist" << std::endl;
	
	else if (ker.row()%2==0 ||ker.col()%2==0)
		std::cerr << "Error in MatrixHybrid::convolve call: kernel dimensions must be odd values" << std::endl;

	else if (ker.row()>nRow || ker.col()>nCol)
		std::cerr << "Error in MatrixHybrid::convolve call: invalid kernel dimension(s)" << std::endl;
	
	else
	{
		const dimType nR = row();
		const dimType nC = col();
		
		MatrixHybrid res(nR-ker.row()+1,nC,proc);	// matrice risultante
		
		#pragma omp parallel
		{
			const dimType kCol=ker.col();
			const dimType shj = kCol/2;
			
			const dimType kFirst = ker.cell()*proc.lRank()/proc.lSize();
			const dimType kLast = ker.cell()*(proc.lRank()+1)/proc.lSize();
			
			dimType ki=0;
			dimType kj=0;
			
			const elemType *const idxThis = reset();
			const elemType *const idxKer = ker.reset();
			elemType *const idxRes = res.reset();
			
			#pragma omp for
			for(dimType i=0; i<res.nRow; ++i)
			{
				for(dimType j=0; j<nC; ++j)
				{
					for(dimType kKer=kFirst; kKer < kLast; ++kKer)
					{
						ki = kKer/kCol;
						kj = kKer%kCol;
						
						idxRes[i*nCol+j] += idxThis[getIndex(i+ki,j+kj-shj,nR,nC)] * idxKer[kKer];	// to be checked
					}
				}
			}
		}
		
		*this = res;
		
		return EXIT_SUCCESS;
	}
	
	return EXIT_FAILURE;
}

inline int MatrixHybrid::convolve(const Matrix ker, const unsigned int iter)
{
	if (iter == 0)
		return EXIT_SUCCESS;
	
	if (proc.wSize() == 1)
		return MatrixOMP::convolve(ker,iter);
	
	const dimType nR = nRow;
	const dimType shi = ker.row()/2;
	
	if (proc.spreadData(*this,ker) == EXIT_FAILURE)
	{
		if (proc.wRank() == 0)
			std::cerr << "Error in MatrixHybrid::convolve call: unable to properly deliver data" << std::endl;
		return EXIT_FAILURE;
	}
	
	if(MatrixHybrid::convolve(ker) == EXIT_FAILURE)
	{
		if (proc.wRank() == 0)
			std::cerr << "Error in MatrixHybrid::convolve at iteration " << 1 << ": unable to perform matrix convolution" << std::endl;
		return EXIT_FAILURE;
	}
	
	for(unsigned int it = 1; it < iter; ++it)
	{
		if (proc.shareData(*this,shi) == EXIT_FAILURE)
		{
			if (proc.wRank() == 0)
				std::cerr << "Error in MatrixHybrid::convolve call: unable to properly share data" << std::endl;
			return EXIT_FAILURE;
		}
		
		if(MatrixHybrid::convolve(ker) == EXIT_FAILURE)
		{
			if (proc.wRank() == 0)
				std::cerr << "Error in MatrixHybrid::convolve at iteration " << it+1 << ": unable to perform matrix convolution" << std::endl;
			return EXIT_FAILURE;
		}
	}
	
	return proc.collectData(*this,nR);
}

#endif
