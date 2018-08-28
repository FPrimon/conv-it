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
#ifndef CONV_IT_CUDA_DEF
#define CONV_IT_CUDA_DEF

#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

#include "conv_it_def.hpp"

const int BlocksPerMultiProcessor = 8;
const int BlocksPerMultiProcessor_v3 = 16;

const size_t elemSize = sizeof(elemType);

const dimType kernelMaxNumberOfCells = 3025;

__constant__ elemType devKer[kernelMaxNumberOfCells];

__host__
inline int handleError(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		std::string strError(cudaGetErrorString(error));
		std::cerr << "CUDA error: " << strError << std::endl;
		
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

__device__
inline dimType getIndexCUDA(signedDimType idx, const signedDimType dim)
{
	if(dim <= 0)
		return 0;
	
	for( ; idx<0 ; idx+=dim);
	for( ; idx>=dim ; idx-=dim);
	
	return static_cast<dimType> (idx);
}

__host__
inline int kerToDevice(const elemType *hostKer, size_t count)
{
	return handleError( cudaMemcpyToSymbol(devKer,hostKer,count,0,cudaMemcpyHostToDevice) );
}

__host__
inline int matToDevice(elemType *devPtr, size_t devPitch, const elemType *hostPtr, size_t width, size_t height)
{
	return handleError( cudaMemcpy2D(devPtr,devPitch,hostPtr,width,width,height,cudaMemcpyHostToDevice) );
}

__host__
inline int matFromDevice(const elemType *devPtr, size_t devPitch, elemType *hostPtr, size_t width, size_t height)
{
	return handleError( cudaMemcpy2D(hostPtr,width,devPtr,devPitch,width,height,cudaMemcpyDeviceToHost) );
}

__global__
void swapMatrix(elemType *const devMat, const elemType *const devRes, const size_t pitch, const unsigned int lastTinB)
{
	if (threadIdx.x < lastTinB)
	{
		const dimType i = blockDim.y*blockIdx.y+threadIdx.y;
		const dimType j = lastTinB*blockIdx.x+threadIdx.x;
		
		//T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
		elemType *const idxMat = (elemType*)((char*)devMat + i*pitch) + j;
		const elemType *const idxRes = (elemType*)((char*)devRes + i*pitch) + j;
		
		*idxMat = *idxRes;
	}
}

__global__
void convolution(const elemType *const devMat, elemType *const devRes, const dimType kRow, const dimType kCol, const size_t pitch, const unsigned int lastTinB)
{
	if (threadIdx.x < lastTinB)
	{
		const dimType nRow = gridDim.y*blockDim.y;
		const dimType nCol = gridDim.x*lastTinB;
		
		const dimType i = blockDim.y*blockIdx.y+threadIdx.y;
		const dimType j = lastTinB*blockIdx.x+threadIdx.x;
		
		const dimType shi = kRow/2;
		const dimType shj = kCol/2;
		
		//T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
		elemType *const idxRes = (elemType*)((char*)devRes + i*pitch) + j;
		const elemType *idxMat = NULL;
		
		*idxRes = 0;
		
		for(dimType ki=0; ki < kRow; ++ki)
		{
			for(dimType kj=0; kj < kCol; ++kj)
			{
				idxMat = (elemType*)((char*)devMat + getIndexCUDA(i+ki-shi,nRow)*pitch) + getIndexCUDA(j+kj-shj,nCol);
				*idxRes += devKer[ki*kCol+kj] * idxMat[0];
				
				//__syncthreads();	// per ottimizare accesso alla constant memory
			}
		}
	}
}

#endif
