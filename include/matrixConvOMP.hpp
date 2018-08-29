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
#ifndef MATRIX_CONV_OMP
#define MATRIX_CONV_OMP

#include <cstdlib>
#include <iostream>
#include <string>
#include <omp.h>

#include "conv_it_def.hpp"
#include "matrixConv.hpp"

//	CLASSes DECLARATIONs
class MatrixOMP;

//	CLASSes DEFINITIONs
class MatrixOMP : virtual public Matrix	// Pointer-like Class
{
	//	CONSTRUCTORs
	public:
		MatrixOMP() = default;		// costruisce la matrice 0x0
		MatrixOMP(const dimType nR, const dimType nC) : Matrix(nR,nC)	{}
		MatrixOMP(const dimType nR, const dimType nC, const elemType *const p) : Matrix(nR,nC,p)	{}
		MatrixOMP(const dimType nR, const dimType nC, std::istream &input) : Matrix(nR,nC,input)	{}
		MatrixOMP(const dimType nR, const dimType nC, const std::string filename) : Matrix(nR,nC,filename)	{}
	
	//	COPY-CONSTRUCTOR
	public:
		MatrixOMP(const MatrixOMP &mat) : Matrix(mat)	{}	// copia gli attributi, ma non alloca ulteriore memoria
		
	//	OPERATORs
	public:
		MatrixOMP& operator=(const MatrixOMP &rightMat)		// copia gli attributi, ma non alloca ulteriore memoria
		{
			Matrix::operator=(rightMat);
			return *this;
		}
	
	//	MEMBER-FUNCTIONs DECLARATIONs
	public:
		int fromDynArr(const dimType nR, const dimType nC, const elemType *const arr)	override;
		
		void toDynArr(elemType *&arr) const	override;
		
		int addRows(const dimType nR, const elemType *const above, const elemType *const below)	override;
		int rowPrepend(const dimType nR, const elemType *const rows)	override;
		int rowAppend(const dimType nR, const elemType *const rows)	override;
		
		int convolve(const Matrix ker)	override;
		int convolve(const Matrix ker, const unsigned int iter) override;
	
	//	DESTRUCTOR
	public:
		~MatrixOMP() = default;
};

//	MEMBER-FUNCTIONs DEFINITIONs
inline int MatrixOMP::fromDynArr(const dimType nR, const dimType nC, const elemType *const arr)
{
	if (arr == nullptr)
	{
		std::cerr << "Error in MatrixOMP::fromDynArr call: array does not exist" << std::endl;
		clear();	// cancello la matrice
		return EXIT_FAILURE;
	}
	
	if (nR<=0 || nC<=0)
	{
		std::cerr << "Error in MatrixOMP::fromDynArr call: non-positive matrix dimension(s)" << std::endl;
		clear();	// cancello la matrice
		return EXIT_FAILURE;
	}
	
	MatrixOMP tmp(nR,nC);
	
	#pragma omp parallel
	{
		elemType *const idxTmp = tmp.idx;
		
		#pragma omp for
		for(dimType k=0; k < tmp.nCell; ++k)
			idxTmp[k] = arr[k];
	}
	
	*this = tmp;
	
	return EXIT_SUCCESS;
}

inline void MatrixOMP::toDynArr(elemType *&arr) const
{
	if (arr != nullptr)
		delete [] arr;	// solo con array allocati dinamicamente, non con oggetti
	
	if (reset() == nullptr)
	{
		arr = nullptr;
		return;
	}
	
	arr = new elemType[nCell];
	
	#pragma omp parallel
	{
		const elemType *const idxThis = idx;
		elemType *const idxArr = arr;
		
		#pragma omp for
		for(dimType k=0; k < nCell; ++k)
			idxArr[k] = idxThis[k];
	}
}

inline int MatrixOMP::addRows(const dimType nR, const elemType *const above, const elemType *const below)
{
	if (above == nullptr || below == nullptr)
	{
		std::cerr << "Error in MatrixOMP::addRows call: array(s) do(es) not exist" << std::endl;
		return EXIT_FAILURE;
	}
	
	if (nR<=0)
	{
		std::cerr << "Error in MatrixOMP::addRows call: non-positive number of rows" << std::endl;
		return EXIT_FAILURE;
	}
	
	MatrixOMP tmp(2*nR+nRow,nCol);
	
	#pragma omp parallel
	{
		elemType *const idxTmp = tmp.idx;
		const elemType *const idxThis = reset();
		
		#pragma omp for
		for(dimType i=0; i < nR; ++i)
		{
			for(dimType j=0; j < nCol; ++j)
			{
				idxTmp[i*nCol+j] = above[i*nCol+j];
				idxTmp[(i+nR+nRow)*nCol+j] = below[i*nCol+j]; 
				//tmp(i,j) = above[i*nCol+j];
				//tmp(i+nR+nRow,j) = below[i*nCol+j];
			}
		}
		
		#pragma omp for
		for(dimType i=0; i < nRow; ++i)
		{
			for(dimType j=0; j < nCol; ++j)
			{
				idxTmp[(i+nR)*nCol+j] = idxThis[i*nCol+j];
				//tmp(i+nR,j) = (*this)(i,j);
			}
		}
	}
	
	*this = tmp;
	
	return EXIT_SUCCESS;
}

inline int MatrixOMP::rowPrepend(const dimType nR, const elemType *const rows)
{
	if (rows == nullptr)
	{
		std::cerr << "Error in MatrixOMP::rowPrepend call: array does not exist" << std::endl;
		return EXIT_FAILURE;
	}
	
	if (nR<=0)
	{
		std::cerr << "Error in MatrixOMP::rowPrepend call: non-positive number of rows" << std::endl;
		return EXIT_FAILURE;
	}
	
	MatrixOMP tmp(nR+nRow,nCol);
	
	#pragma omp parallel
	{
		elemType *const idxTmp = tmp.idx;
		const elemType *const idxThis = reset();
		
		#pragma omp for
		for(dimType i=0; i < nR; ++i)
		{
			for(dimType j=0; j < nCol; ++j)
			{
				idxTmp[i*nCol+j] = rows[i*nCol+j];
				//tmp(i,j) = rows[i*nCol+j];
			}
		}
		
		#pragma omp for
		for(dimType i=0; i < nRow; ++i)
		{
			for(dimType j=0; j < nCol; ++j)
			{
				idxTmp[(i+nR)*nCol+j] = idxThis[i*nCol+j];
				//tmp(i+nR,j) = (*this)(i,j);
			}
		}
	}
	
	*this = tmp;
	
	return EXIT_SUCCESS;
}

inline int MatrixOMP::rowAppend(const dimType nR, const elemType *const rows)
{
	if (rows == nullptr)
	{
		std::cerr << "Error in MatrixOMP::rowAppend call: array does not exist" << std::endl;
		return EXIT_FAILURE;
	}
	
	if (nR<=0)
	{
		std::cerr << "Error in MatrixOMP::rowAppend call: non-positive number of rows" << std::endl;
		return EXIT_FAILURE;
	}
	
	MatrixOMP tmp(nRow+nR,nCol);
	
	#pragma omp parallel
	{
		elemType *const idxTmp = tmp.idx;
		const elemType *const idxThis = reset();
		
		#pragma omp for
		for(dimType i=0; i < nR; ++i)
		{
			for(dimType j=0; j < nCol; ++j)
			{
				idxTmp[(i+nRow)*nCol+j] = rows[i*nCol+j];
				//tmp(i+nRow,j) = rows[i*nCol+j];
			}
		}
		
		#pragma omp for
		for(dimType i=0; i < nRow; ++i)
		{
			for(dimType j=0; j < nCol; ++j)
			{
				idxTmp[i*nCol+j] = idxThis[i*nCol+j];
				//tmp(i,j) = (*this)(i,j);
			}
		}
	}
	
	*this = tmp;
	
	return EXIT_SUCCESS;
}

inline int MatrixOMP::convolve(const Matrix ker)
{
	if (reset() == nullptr)
		std::cerr << "Error in MatrixOMP::convolve call: matrix does not exist" << std::endl;
	
	else if (ker.reset() == nullptr)
		std::cerr << "Error in MatrixOMP::convolve call: kernel does not exist" << std::endl;
	
	else if (ker.row()%2==0 ||ker.col()%2==0)
		std::cerr << "Error in MatrixOMP::convolve call: kernel dimensions must be odd values" << std::endl;

	else if (ker.row()>nRow || ker.col()>nCol)
		std::cerr << "Error in MatrixOMP::convolve call: invalid kernel dimension(s)" << std::endl;
	
	else
	{
		MatrixOMP res(nRow,nCol);	// matrice risultante
		
		#pragma omp parallel
		{
			const dimType kRow = ker.row();
			const dimType kCol = ker.col();
			const dimType shi = kRow/2;
			const dimType shj = kCol/2;
			
			const elemType *const idxThis = idx;
			const elemType *const idxKer = ker.reset();
			elemType *const idxRes = res.idx;
			
			#pragma omp for
			for(dimType i=0; i<nRow; ++i)
			{
				for(dimType j=0; j<nCol; ++j)
				{
					for(dimType ki=0; ki<kRow; ++ki)
					{
						for(dimType kj=0; kj<kCol; ++kj)
						{
							idxRes[i*nCol+j] += idxThis[getIndex(i+ki-shi,j+kj-shj,nRow,nCol)] * idxKer[ki*kCol+kj];
						}
					}
				}
			}
		}
		
		/*
		#pragma omp parallel
		{
			dimType kRow = ker.row();
			dimType kCol = ker.col();
			dimType shi = kRow/2;
			dimType shj = kCol/2;
			
			elemType *idxThis = reset();
			elemType *idxKer = ker.reset();
			elemType *idxRes = res.reset();
			
			#pragma omp for
			for(dimType k=0; k<nCell; ++k)
			{
				dimType i = k/nCol;
				dimType j = k%nCol;
				
				for(dimType kKer=0; kKer<ker.cell(); ++kKer)
				{
					dimType ki = kKer/kCol;
					dimType kj = kKer%kCol;
					
					idxRes[k] += idxThis[getIndex(i+ki-shi,j+kj-shj,nRow,nCol)] * idxKer[kKer];
				}
			}
		}
		*/
		
		*this = res;
		
		return EXIT_SUCCESS;
	}
	
	return EXIT_FAILURE;
}

inline int MatrixOMP::convolve(const Matrix ker, const unsigned int iter)
{
	if (iter == 0)
		return EXIT_SUCCESS;
	
	for(unsigned int it = 0; it < iter; ++it)
	{
		if(MatrixOMP::convolve(ker) == EXIT_FAILURE)
		{
			std::cerr << "Error in MatrixOMP::convolve at iteration " << it+1 << ": unable to perform matrix convolution" << std::endl;
			return EXIT_FAILURE;
		}
	}
	
	return EXIT_SUCCESS;
}

#endif
