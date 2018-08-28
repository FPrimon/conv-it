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
#ifndef MATRIX_CONV_CUDA
#define MATRIX_CONV_CUDA

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

#include "conv_it_def.hpp"
#include "conv_it_cuda_def.hpp"

//	CLASSes DECLARATION
class CUDAdev;
class MatrixCUDA;

//	CLASSes DEFINITIONs
class CUDAdev
{
	//	FRIENDs DECLARATIONs
	friend class MatrixCUDA;
	
	//	MEMBERs DEFINITIONs
	private:
		unsigned int devID;
		
		unsigned int warpSize;
		unsigned int maxThreadspBlock;
		unsigned int maxThreadspMultiProcessor;
		
		unsigned int maxWarpspMultiProcessor;
		unsigned int maxBlockspMultiProcessor;
		
		dim3 maxBlockSize;
		dim3 maxGridSize;
		
		dim3 blockSize;
		dim3 gridSize;
		
		unsigned int warps;
		unsigned int lastThreadinBlock;
		
	//	CONSTRUCTORs
	public:
		CUDAdev();
	
	//	COPY-CONSTRUCTOR
	public:
		
	//	MEMBER-FUNCTIONs DECLARATIONs
	public:
		void printInfo() const;
		void check() const;
		int setDevice(const int dev);
		int kernelSetup(MatrixCUDA mat);
	
	//	MEMBER-FUNCTIONs DECLARATIONs
	public:
		
	//	DESTRUCTOR
	public:
		~CUDAdev()	{}
};

class MatrixCUDA	// Pointer-like Class
{
	//	TYPEs DEFINITIONs
	public:
		//typedef long int signedDimType;
		//typedef unsigned long int dimType;
		//typedef long int elemType;
	
	//	FRIENDs DECLARATION
	
	//	MEMBERs DEFINITIONs
	protected:
		CUDAdev device;
		elemType* devMat;
		elemType* devRes;
		
		dimType nRow;		// numero di righe
		dimType nCol;		// numero di colonne
		dimType nCell;	// nRow*nCol
		
		elemType *idx;	// indice di scorrimento
		void *begin;	// inizio dell'array
		void *end;		// fine dell'array
	private:
		size_t *busy;		// reference count
	
	//	CONSTRUCTORs DECLARATIONs
	public:
		MatrixCUDA() : device(), devMat(NULL), devRes(NULL), nRow(0), nCol(0), nCell(0), idx(NULL), begin(NULL), end(NULL), busy(NULL) {}	// costruisce la matrice 0x0
		MatrixCUDA(const dimType nR, const dimType nC);		// costruisce la matrice NULLA nRxnC
		MatrixCUDA(const dimType nR, const dimType nC, const std::string filename);	// costruisce la matrice da file testuale
	
	//	COPY-CONSTRUCTOR
	public:
		MatrixCUDA(const MatrixCUDA &mat):		// copia gli attributi, ma non alloca ulteriore memoria
			device(mat.device), devMat(mat.devMat), devRes(mat.devRes), nRow(mat.nRow), nCol(mat.nCol), nCell(mat.nCell), idx(mat.idx), begin(mat.begin), end(mat.end), busy(mat.busy)
		{
			if(busy != NULL)		// se non sto copiando la matrice 0x0
				++*busy;
		}
	
	//	MEMBER-FUNCTIONs DEFINITIONs
	public:
		elemType* reset()	// riporta l'indice all'inizio e restituisce il valore iniziale
		{
			idx = static_cast<elemType*> (begin);
			return idx;
		}
		
		const elemType* reset() const	{ return static_cast<elemType*> (begin); }
		
		dimType row() const	{ return nRow; }
		dimType col() const	{ return nCol; }
		dimType cell() const	{ return nCell; }
		
		int kernelSetup()	{ return device.kernelSetup(*this); }
		void check()	{ device.check(); }
		
	//	OPERATORs DECLARATIONs
	public:
		MatrixCUDA& operator=(const MatrixCUDA &rightMat);	// copia gli attributi, ma non alloca ulteriore memoria
		elemType& operator()(const signedDimType i, const signedDimType j);	// subscript + wrap around
		const elemType& operator()(const signedDimType i, const signedDimType j) const;
	
	//	MEMBER-FUNCTIONs DECLARATIONs
	public:
		void clear();	// cancella la matrice, si ottiene una matrice 0x0
		
		int fromStream(const dimType nR, const dimType nC, std::istream &input);
		int fromFile(const dimType nR, const dimType nC, const std::string filename);
		
		void toStream(std::ostream &output) const;
		void toFile(const std::string filename) const;
	
	protected:
		int convolve(const MatrixCUDA ker, const size_t pitch);
	public:
		int convolve(const MatrixCUDA ker, const unsigned int iter);	// overloaded functions: tutte o nessuna
	
	//	DESTRUCTOR
	public:
		~MatrixCUDA()
		{
			if (busy!=NULL && --*busy == 0)	// se è l'ultimo oggetto tiene occupata la memoria allocata
			{
				reset();
				delete [] idx;	// libero la memoria allocata
				delete busy;	// libero il contatore
			}
		}
};

//	CONSTRUCTORs DEFINITIONs
inline CUDAdev::CUDAdev():
	devID(0), warpSize(0), maxThreadspBlock(0), maxThreadspMultiProcessor(0), maxWarpspMultiProcessor(0), maxBlockspMultiProcessor(0), warps(0), lastThreadinBlock(0)
{
	int deviceCount;
	if (handleError(cudaGetDeviceCount(&deviceCount)) == EXIT_FAILURE)
		std::cerr << "Error in CUDAdev constructor: cudaGetDeviceCount failed" << std::endl;
	
	if (deviceCount == 0)
		std::cerr << "Error in CUDAdev constructor: no CUDA capable device found" << std::endl;
	
	else
	{
		int major=0;
		int minor=0;
		cudaDeviceProp prop;
		
		for(int dev=0; dev < deviceCount; ++dev)
		{
			if (handleError(cudaGetDeviceProperties(&prop,dev)) == EXIT_FAILURE)
				std::cerr << "Error in CUDAdev constructor: cudaGetDeviceProperties failed" << std::endl;
			
			if (prop.major >= major && prop.minor > minor)
			{
				major = prop.major;
				minor = prop.minor;
				
				devID = dev;
			}
		}
		
		if (setDevice(devID) == EXIT_FAILURE)
			std::cerr << "Error in CUDAdev constructor: setDevice failed" << std::endl;
	}
}

inline MatrixCUDA::MatrixCUDA(const dimType nR, const dimType nC):		// matrice inizializzata a 0, busy inizializzato a 1
	device(), devMat(NULL), devRes(NULL), nRow(nR), nCol(nC), nCell(nR*nC), idx(new elemType[nCell]()), begin(idx), end(idx+nCell), busy(new size_t(1))
{
	if (nCell == 0)
		clear();	// cancello la matrice
}

inline MatrixCUDA::MatrixCUDA(const dimType nR, const dimType nC, const std::string filename):
	device(), devMat(NULL), devRes(NULL), nRow(0), nCol(0), nCell(0), idx(NULL), begin(NULL), end(NULL), busy(NULL)
{
	if (fromFile(nR,nC,filename) == EXIT_FAILURE)
		std::cerr << "Error in Matrix constructor call: unable to construct matrix" << std::endl;
}

//	OPERATORs DEFINITIONs
inline MatrixCUDA& MatrixCUDA::operator=(const MatrixCUDA &rightMat)
{
	if (rightMat.busy != NULL)	// se non sto copiando la matrice 0x0
		++*rightMat.busy;
	
	if (busy!=NULL && --*busy == 0)	// se è l'ultimo oggetto tiene occupata la memoria allocata
	{
		reset();
		delete [] idx;	// libero la memoria allocata
		delete busy;	// libero il contatore
	}
	
	busy = rightMat.busy;
	end = rightMat.end;
	begin = rightMat.begin;
	idx = rightMat.idx;
	
	nCell = rightMat.nCell;
	nCol = rightMat.nCol;
	nRow = rightMat.nRow;
	
	devRes = rightMat.devRes;
	devMat = rightMat.devMat;
	device = rightMat.device;
	
	return *this;
}

inline elemType& MatrixCUDA::operator()(const signedDimType i, const signedDimType j)	// solo se nCell !=0
{
	reset();
	return idx[getIndex(i,j,nRow,nCol)];
}

inline const elemType& MatrixCUDA::operator()(const signedDimType i, const signedDimType j) const	// solo se nCell !=0
{
	return idx[getIndex(i,j,nRow,nCol)];
}

//	MEMBER-FUNCTIONs DEFINITIONs
inline void CUDAdev::printInfo() const
{
	std::cout << "devID = " << devID << std::endl;
	std::cout << "warpSize = " << warpSize << std::endl;
	std::cout << "maxThreadspBlock = " << maxThreadspBlock << std::endl;
	std::cout << "maxThreadspMultiProcessor = " << maxThreadspMultiProcessor << std::endl;
	std::cout << "maxWarpspMultiProcessor = " << maxWarpspMultiProcessor << std::endl;
	std::cout << "maxBlockspMultiProcessor = " << maxBlockspMultiProcessor << std::endl;
	std::cout << "maxBlockSize = (" << maxBlockSize.x << "," << maxBlockSize.y << "," << maxBlockSize.z << ")" << std::endl;
	std::cout << "maxGridSize = (" << maxGridSize.x << "," << maxGridSize.y << "," << maxGridSize.z << ")" << std::endl;
}

inline void CUDAdev::check() const
{
	printInfo();
	
	std::cout << "blockSize = (" << blockSize.x << "," << blockSize.y << "," << blockSize.z << ")" << std::endl;
	std::cout << "gridSize = (" << gridSize.x << "," << gridSize.y << "," << gridSize.z << ")" << std::endl;
	std::cout << "warps = " << warps << std::endl;
	std::cout << "lastThreadinBlock = " << lastThreadinBlock << std::endl;
}

inline int CUDAdev::setDevice(const int dev)
{
	devID = dev;
	
	if (handleError(cudaSetDevice(dev)) == EXIT_FAILURE)
	{
		std::cerr << "Error in CUDAdev constructor: cudaSetDevice failed" << std::endl;
		return EXIT_FAILURE;
	}
	
	cudaDeviceProp prop;
	if (handleError(cudaGetDeviceProperties(&prop,dev)) == EXIT_FAILURE)
	{
		std::cerr << "Error in CUDAdev constructor: cudaGetDeviceProperties failed" << std::endl;
		return EXIT_FAILURE;
	}
	
	warpSize = prop.warpSize;
	maxThreadspBlock = prop.maxThreadsPerBlock;
	maxThreadspMultiProcessor = prop.maxThreadsPerMultiProcessor;
	
	maxWarpspMultiProcessor = maxThreadspMultiProcessor/warpSize;
	
	const int major = prop.major;
	
	if (major > 2)
		maxBlockspMultiProcessor = BlocksPerMultiProcessor_v3;
	else
		maxBlockspMultiProcessor = BlocksPerMultiProcessor;
	
	maxBlockSize.x = prop.maxThreadsDim[0];
	maxBlockSize.y = prop.maxThreadsDim[1];
	maxBlockSize.z = 1;
	
	maxGridSize.x = prop.maxGridSize[0];
	maxGridSize.y = prop.maxGridSize[1];
	maxGridSize.z = 1;
	
	return EXIT_SUCCESS;
}

inline int CUDAdev::kernelSetup(MatrixCUDA mat)
{
	if (warpSize == 0)
	{
		std::cerr << "Error in CUDAdev::kernelSetup: uninitialized device" << std::endl;
		return EXIT_FAILURE;
	}
	
	const dimType nR=mat.row();
	dimType nC=mat.col();
	
	if (nC <= maxThreadspBlock)
	{
		gridSize.x = 1;		// quante colonne di blocchi nella griglia?
		
		lastThreadinBlock = nC;
	}
	else
	{
		unsigned int x;
		for(x=2; nC/x > maxThreadspBlock || nC%x != 0; ++x);
		
		gridSize.x = x;		// quante colonne di blocchi nella griglia?
		
		lastThreadinBlock = nC/x;
	}
	
	blockSize.x = lastThreadinBlock;
	
	unsigned int residue = blockSize.x%warpSize;
	if (residue != 0)
		blockSize.x += warpSize-residue;
	
	unsigned int actualMaxBlockSize_y = maxThreadspBlock/blockSize.x;
	
	if (nR <= actualMaxBlockSize_y)
	{
		gridSize.y = 1;		// quante righe di blocchi nella griglia?
		blockSize.y = nR;
	}
	else
	{
		unsigned int y;
		for(y=2; nR/y > actualMaxBlockSize_y || nR%y != 0; ++y);
		
		gridSize.y = y;		// quante righe di blocchi nella griglia?
		blockSize.y = nR/y;
	}
	
	if (blockSize.x > maxBlockSize.x || blockSize.y > maxBlockSize.y || blockSize.x*blockSize.y > maxThreadspBlock)
	{
		std::cerr << "Error in CUDAdev::kernelSetup: blocks created were too large" << std::endl;
		return EXIT_FAILURE;
	}
	
	if (gridSize.x > maxGridSize.x || gridSize.y > maxGridSize.y)
	{
		std::cerr << "Error in CUDAdev::kernelSetup: grid created was too large" << std::endl;
		return EXIT_FAILURE;
	}
	
	blockSize.z=1;
	gridSize.z=1;
	
	warps = blockSize.x*blockSize.y*std::min(gridSize.x*gridSize.y,maxBlockspMultiProcessor)/warpSize;
	
	if (warps < maxWarpspMultiProcessor)
		std::cerr << "Warning in CUDAdev::kernelSetup: underutilized resources - only " << warps << " simultaneously residing warps (maximum is " << maxWarpspMultiProcessor << ")" << std::endl;
	
	return EXIT_SUCCESS;
}

inline void MatrixCUDA::clear()
{
	if (busy!=NULL && --*busy == 0)	// se è l'ultimo oggetto tiene occupata la memoria allocata
	{
		reset();
		delete [] idx;	// libero la memoria allocata
		delete busy;	// libero il contatore
	}
	
	busy = NULL;
	end = NULL;
	begin = NULL;
	idx = NULL;
	
	nCell = 0;
	nCol = 0;
	nRow = 0;
}

inline int MatrixCUDA::fromStream(const dimType nR, const dimType nC, std::istream &input)
{
	if (nR<=0 || nC<=0)
	{
		std::cerr << "Error in Matrix::fromStream call: non-positive matrix dimension(s)" << std::endl;
		clear();	// cancello la matrice
		return EXIT_FAILURE;
	}
	
	std::istream::iostate oldState=input.rdstate();
	input.clear();
	
	MatrixCUDA tmp(nR,nC);
	
	dimType kR=0;
	for(std::string line; getline(input,line) && kR<tmp.nRow; ++kR)	// ordine delle condizioni è fondamentale
	{
		dimType kC=0;
		std::istringstream buffer(line);
		
		for( ; buffer>>tmp.idx[kR*tmp.nCol+kC] && kC<tmp.nCol; ++kC);	// ordine delle condizioni è fondamentale
		
		if(!buffer.eof() || kC!=tmp.nCol)
		{
			std::cerr << "Error in Matrix::fromStream call: invalid number of columns" << std::endl;
			clear();
			input.setstate(oldState);
			return EXIT_FAILURE;
		}
	}
	
	if(!input.eof() || kR!=tmp.nRow)
	{
		std::cerr << "Error in Matrix::fromStream call: invalid number of rows" << std::endl;
		clear();
		input.setstate(oldState);
		return EXIT_FAILURE;
	}
	
	*this = tmp;
	
	input.setstate(oldState);
	return EXIT_SUCCESS;
}

inline int MatrixCUDA::fromFile(const dimType nR, const dimType nC, const std::string file)
{
	const char *filename = file.c_str();
	std::ifstream inputFile(filename);
	return fromStream(nR,nC,inputFile);
}

inline void MatrixCUDA::toStream(std::ostream &output) const
{
	if (nCell == 0)
		return;
	
	std::istream::iostate oldState=output.rdstate();
	output.clear();
	
	for(dimType i=0; i < nRow; ++i)
	{
		for(dimType j=0; j < nCol-1; ++j)
		{
			output << (*this)(i,j) << " ";
			//output << static_cast<int> ((*this)(i,j)) << " ";
		}
		
		output << (*this)(i,nCol-1) << std::endl;
		//output << static_cast<int> ((*this)(i,nCol-1)) << std::endl;
	}
	
	// output << std::flush;	// superfluo
	output.setstate(oldState);
}

inline void MatrixCUDA::toFile(const std::string file) const
{
	const char *filename = file.c_str();
	std::ofstream outputFile(filename);
	toStream(outputFile);
}

inline int MatrixCUDA::convolve(const MatrixCUDA ker, size_t pitch)
{
	if (reset() == NULL)
		std::cerr << "Error in MatrixCUDA::convolve call: matrix does not exist" << std::endl;
	
	else if (ker.reset() == NULL)
		std::cerr << "Error in MatrixCUDA::convolve call: kernel does not exist" << std::endl;
	
	else if (ker.row()%2==0 ||ker.col()%2==0)
		std::cerr << "Error in MatrixCUDA::convolve call: kernel dimensions must be odd values" << std::endl;

	else if (ker.row()>nRow || ker.col()>nCol)
		std::cerr << "Error in MatrixCUDA::convolve call: invalid kernel dimension(s)" << std::endl;
	
	else
	{
		if (handleError(cudaMallocPitch(&devRes,&pitch,nCol*elemSize,nRow)) == EXIT_FAILURE)
		{
			std::cerr << "Error in MatrixCUDA::convolve: cudaMallocPitch on devRes failed" << std::endl;
			handleError(cudaFree(devMat));
			return EXIT_FAILURE;
		}
		
		convolution<<<device.gridSize,device.blockSize>>>(devMat,devRes,ker.nRow,ker.nCol,pitch,device.lastThreadinBlock);
		
		swapMatrix<<<device.gridSize,device.blockSize>>>(devMat,devRes,pitch,device.lastThreadinBlock);
		
		handleError(cudaFree(devRes));
		
		return EXIT_SUCCESS;
	}
	
	return EXIT_FAILURE;
}

inline int MatrixCUDA::convolve(const MatrixCUDA ker, const unsigned int iter)
{
	if (iter == 0)
		return EXIT_SUCCESS;
	
	if (kernelSetup() == EXIT_FAILURE)
	{
		std::cerr << "Error in MatrixCUDA::convolve: unable to perform kernel setup" << std::endl;
		return EXIT_FAILURE;
	}
	
	//device.check();
		
	size_t pitch;
	
	if (handleError(cudaMallocPitch(&devMat,&pitch,nCol*elemSize,nRow)) == EXIT_FAILURE)
	{
		std::cerr << "Error in MatrixCUDA::convolve: cudaMallocPitch on devMat failed" << std::endl;
		return EXIT_FAILURE;
	}
	
	if (matToDevice(devMat,pitch,reset(),nCol*elemSize,nRow) == EXIT_FAILURE)
	{
		std::cerr << "Error in MatrixCUDA::convolve: matToDevice failed" << std::endl;
		handleError(cudaFree(devMat));
		return EXIT_FAILURE;
	}
	
	if (kerToDevice(ker.reset(),elemSize*ker.cell()) == EXIT_FAILURE)
	{
		std::cerr << "Error in MatrixCUDA::convolve: kerToDevice failed" << std::endl;
		handleError(cudaFree(devMat));
		return EXIT_FAILURE;
	}
	
	for(unsigned int it = 0; it < iter; ++it)
	{
		if(MatrixCUDA::convolve(ker,pitch) == EXIT_FAILURE)
		{
			std::cerr << "Error in MatrixCUDA::convolve at iteration " << it+1 << ": unable to perform matrix convolution" << std::endl;
			handleError(cudaFree(devMat));
			return EXIT_FAILURE;
		}
	}
	
	if (matFromDevice(devMat,pitch,reset(),nCol*elemSize,nRow) == EXIT_FAILURE)
	{
		std::cerr << "Error in MatrixCUDA::convolve: matFromDevice failed" << std::endl;
		handleError(cudaFree(devMat));
		return EXIT_FAILURE;
	}

	handleError(cudaFree(devMat));	//--->	corrupted double-linked list ??!?
	
	//handleError(cudaDeviceReset());	//---> segfault ??!?
	
	return EXIT_SUCCESS;
}

#endif
