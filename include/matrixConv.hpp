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
#ifndef MATRIX_CONV
#define MATRIX_CONV

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "conv_it_def.hpp"

//	TYPEs DEFINITIONs

//	CLASSes DECLARATIONs
class Matrix;

//	FUNCTIONs DECLARATIONs

//	CLASSes DEFINITIONs
class Matrix	// Pointer-like Class
{
	//	TYPEs DEFINITIONs
	public:
		//typedef long int signedDimType;
		//typedef unsigned long int dimType;
		//typedef long int elemType;
	
	//	FRIENDs DECLARATION
	
	//	MEMBERs DEFINITIONs
	protected:
		dimType nRow=0;		// numero di righe
		dimType nCol=0;		// numero di colonne
		dimType nCell=0;	// nRow*nCol
		
		elemType *idx=nullptr;	// indice di scorrimento
		void *begin=nullptr;	// inizio dell'array
		void *end=nullptr;		// fine dell'array
	private:
		size_t *busy=nullptr;		// reference count
	
	//	CONSTRUCTORs DECLARATIONs
	public:
		Matrix() = default;		// costruisce la matrice 0x0
		Matrix(const dimType nR, const dimType nC);		// costruisce la matrice NULLA nRxnC
		Matrix(const dimType nR, const dimType nC, const elemType *const p);		// costruisce la matrice da un array dinamico
		Matrix(const dimType nR, const dimType nC, std::istream &input);	// costruisce la matrice da flusso di input
		Matrix(const dimType nR, const dimType nC, const std::string filename);	// costruisce la matrice da file testuale
	
	//	COPY-CONSTRUCTOR
	public:
		Matrix(const Matrix &mat):		// copia gli attributi, ma non alloca ulteriore memoria
			nRow(mat.nRow), nCol(mat.nCol), nCell(mat.nCell), idx(mat.idx), begin(mat.begin), end(mat.end), busy(mat.busy)
		{
			if(busy != nullptr)		// se non sto copiando la matrice 0x0
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
		
	//	OPERATORs DECLARATIONs
	public:
		Matrix& operator=(const Matrix &rightMat);	// copia gli attributi, ma non alloca ulteriore memoria
		elemType& operator()(const signedDimType i, const signedDimType j);	// subscript + wrap around
		const elemType& operator()(const signedDimType i, const signedDimType j) const;
	
	//	MEMBER-FUNCTIONs DECLARATIONs
	public:
		void clear();	// cancella la matrice, si ottiene una matrice 0x0
		
		virtual int fromDynArr(const dimType nR, const dimType nC, const elemType *const arr);
		int fromStream(const dimType nR, const dimType nC, std::istream &input);
		int fromFile(const dimType nR, const dimType nC, const std::string filename);
		
		virtual void toDynArr(elemType *&arr) const;
		void toStream(std::ostream &output) const;
		void toFile(const std::string filename) const;
		
		virtual int addRows(const dimType nR, const elemType *const above, const elemType *const below);
		virtual int rowPrepend(const dimType nR, const elemType *const rows);
		virtual int rowAppend(const dimType nR, const elemType *const rows);
		
		void rwShrink(const dimType nR);
		
		virtual int convolve(const Matrix ker);
		virtual int convolve(const Matrix ker, const unsigned int iter);	// overloaded functions: tutte o nessuna
	
	//	DESTRUCTOR
	public:
		virtual ~Matrix()
		{
			if (busy!=nullptr && --*busy == 0)	// se è l'ultimo oggetto tiene occupata la memoria allocata
			{
				reset();
				delete [] idx;	// libero la memoria allocata
				delete busy;	// libero il contatore
			}
		}
};

//	CONSTRUCTORs DEFINITIONs
inline Matrix::Matrix(const dimType nR, const dimType nC):		// matrice inizializzata a 0, busy inizializzato a 1
	nRow(nR), nCol(nC), nCell(nR*nC), idx(new elemType[nCell]()), begin(idx), end(idx+nCell), busy(new size_t(1))
{
	if (nCell == 0)
		clear();	// cancello la matrice
}

inline Matrix::Matrix(const dimType nR, const dimType nC, const elemType *p)		// matrice inizializzata di default
{
	if (fromDynArr(nR,nC,p) == EXIT_FAILURE)
		std::cerr << "Error in Matrix constructor call: unable to construct matrix" << std::endl;
}

inline Matrix::Matrix(const dimType nR, const dimType nC, std::istream &input)		// matrice inizializzata di default
{
	if (fromStream(nR,nC,input) == EXIT_FAILURE)
		std::cerr << "Error in Matrix constructor call: unable to construct matrix" << std::endl;
}
		
inline Matrix::Matrix(const dimType nR, const dimType nC, const std::string filename)		// matrice inizializzata di default
{
	if (fromFile(nR,nC,filename) == EXIT_FAILURE)
		std::cerr << "Error in Matrix constructor call: unable to construct matrix" << std::endl;
}

//	OPERATORs DEFINITIONs
inline Matrix& Matrix::operator=(const Matrix &rightMat)
{
	if (rightMat.busy != nullptr)	// se non sto copiando la matrice 0x0
		++*rightMat.busy;
	
	if (busy!=nullptr && --*busy == 0)	// se è l'ultimo oggetto tiene occupata la memoria allocata
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
	
	return *this;
}

inline elemType& Matrix::operator()(const signedDimType i, const signedDimType j)	// solo se nCell !=0
{
	reset();
	return idx[getIndex(i,j,nRow,nCol)];
}

inline const elemType& Matrix::operator()(const signedDimType i, const signedDimType j) const	// solo se nCell !=0
{
	return idx[getIndex(i,j,nRow,nCol)];
}

//	MEMBER-FUNCTIONs DEFINITIONs
inline void Matrix::clear()
{
	if (busy!=nullptr && --*busy == 0)	// se è l'ultimo oggetto tiene occupata la memoria allocata
	{
		reset();
		delete [] idx;	// libero la memoria allocata
		delete busy;	// libero il contatore
	}
	
	busy = nullptr;
	end = nullptr;
	begin = nullptr;
	idx = nullptr;
	
	nCell = 0;
	nCol = 0;
	nRow = 0;
}

inline int Matrix::fromDynArr(const dimType nR, const dimType nC, const elemType *const arr)
{
	if (arr == nullptr)
	{
		std::cerr << "Error in Matrix::fromDynArr call: array does not exist" << std::endl;
		clear();	// cancello la matrice
		return EXIT_FAILURE;
	}
	
	if (nR<=0 || nC<=0)
	{
		std::cerr << "Error in Matrix::fromDynArr call: non-positive matrix dimension(s)" << std::endl;
		clear();	// cancello la matrice
		return EXIT_FAILURE;
	}
	
	Matrix tmp(nR,nC);
	
	for(dimType k=0; k < tmp.nCell; ++k)
		tmp.idx[k] = arr[k];
	
	*this = tmp;
	
	return EXIT_SUCCESS;
}

inline int Matrix::fromStream(const dimType nR, const dimType nC, std::istream &input)
{
	if (nR<=0 || nC<=0)
	{
		std::cerr << "Error in Matrix::fromStream call: non-positive matrix dimension(s)" << std::endl;
		clear();	// cancello la matrice
		return EXIT_FAILURE;
	}
	
	std::istream::iostate oldState=input.rdstate();
	input.clear();
	
	Matrix tmp(nR,nC);
	
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
	
	/*
	dimType k=0;
	for( ; input>>tmp.idx[k] && k<tmp.nCell; ++k);	// ordine delle condizioni è fondamentale
	
	if(!input.eof() || k!=tmp.nCell)
	{
		std::cerr << "Error in Matrix::fromStream call: invalid matrix dimension(s)" << std::endl;
		
		std::cerr << "k = " << k << std::endl;
		std::cerr << "input.eof = " << input.eof() << std::endl;
		std::cerr << "input.good = " << input.good() << std::endl;
		std::cerr << "input.bad = " << input.bad() << std::endl;
		std::cerr << "input.fail = " << input.fail() << std::endl;
		
		clear();
		
		input.setstate(oldState);
		return EXIT_FAILURE;
	}
	*/
	
	*this = tmp;
	
	input.setstate(oldState);
	return EXIT_SUCCESS;
}

inline int Matrix::fromFile(const dimType nR, const dimType nC, const std::string filename)
{
	std::ifstream inputFile(filename);
	return fromStream(nR,nC,inputFile);
}

inline void Matrix::toDynArr(elemType *&arr) const
{
	if (arr != nullptr)
		delete [] arr;	// solo con array allocati dinamicamente, non con oggetti
	
	if (reset() == nullptr)
	{
		arr = nullptr;
		return;
	}
	
	arr = new elemType[nCell];
	
	for(dimType k=0; k < nCell; ++k)
		arr[k] = idx[k];
}

inline void Matrix::toStream(std::ostream &output) const
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
		}
		
		output << (*this)(i,nCol-1) << std::endl;
	}
	
	// output << std::flush;	// superfluo
	output.setstate(oldState);
}

inline void Matrix::toFile(const std::string filename) const
{
	std::ofstream outputFile(filename);
	toStream(outputFile);
}

inline int Matrix::addRows(const dimType nR, const elemType *const above, const elemType *const below)
{
	if (above == nullptr || below == nullptr)
	{
		std::cerr << "Error in Matrix::addRows call: array(s) do(es) not exist" << std::endl;
		return EXIT_FAILURE;
	}
	
	if (nR<=0)
	{
		std::cerr << "Error in Matrix::addRows call: non-positive number of rows" << std::endl;
		return EXIT_FAILURE;
	}
	
	Matrix tmp(2*nR+nRow,nCol);
	
	for(dimType i=0; i < nR; ++i)
	{
		for(dimType j=0; j < nCol; ++j)
		{
			tmp(i,j) = above[i*nCol+j];
			tmp(i+nR+nRow,j) = below[i*nCol+j];
		}
	}
	
	for(dimType i=0; i < nRow; ++i)
	{
		for(dimType j=0; j < nCol; ++j)
		{
			tmp(i+nR,j) = (*this)(i,j);
		}
	}
	
	*this = tmp;
	
	return EXIT_SUCCESS;
}

inline int Matrix::rowPrepend(const dimType nR, const elemType *const rows)
{
	if (rows == nullptr)
	{
		std::cerr << "Error in Matrix::rowPrepend call: array does not exist" << std::endl;
		return EXIT_FAILURE;
	}
	
	if (nR<=0)
	{
		std::cerr << "Error in Matrix::rowPrepend call: non-positive number of rows" << std::endl;
		return EXIT_FAILURE;
	}
	
	Matrix tmp(nR+nRow,nCol);
	
	for(dimType i=0; i < nR; ++i)
	{
		for(dimType j=0; j < nCol; ++j)
		{
			tmp(i,j) = rows[i*nCol+j];
		}
	}
	
	for(dimType i=0; i < nRow; ++i)
	{
		for(dimType j=0; j < nCol; ++j)
		{
			tmp(i+nR,j) = (*this)(i,j);
		}
	}
	
	*this = tmp;
	
	return EXIT_SUCCESS;
}

inline int Matrix::rowAppend(const dimType nR, const elemType *const rows)
{
	if (rows == nullptr)
	{
		std::cerr << "Error in Matrix::rowAppend call: array does not exist" << std::endl;
		return EXIT_FAILURE;
	}
	
	if (nR<=0)
	{
		std::cerr << "Error in Matrix::rowAppend call: non-positive number of rows" << std::endl;
		return EXIT_FAILURE;
	}
	
	Matrix tmp(nRow+nR,nCol);
	
	for(dimType i=0; i < nR; ++i)
	{
		for(dimType j=0; j < nCol; ++j)
		{
			tmp(i+nRow,j) = rows[i*nCol+j];
		}
	}
	
	for(dimType i=0; i < nRow; ++i)
	{
		for(dimType j=0; j < nCol; ++j)
		{
			tmp(i,j) = (*this)(i,j);
		}
	}
	
	*this = tmp;
	
	return EXIT_SUCCESS;
}

inline void Matrix::rwShrink(const dimType nR)
{
	if (nR<=0)
		std::cerr << "Warning in Matrix::rwShrink call: non-positive number of rows \t Ignored" << std::endl;
		
	else if (2*nR >= nRow)
	{
		std::cerr << "Warning in Matrix::rwShrink call: number of rows exceeds matrix dimension \t Matrix cleared" << std::endl;
		clear();
	}
	else if(fromDynArr(nRow-2*nR,nCol,reset()+nR*nCol) == EXIT_FAILURE)
		std::cerr << "Error in Matrix::rwShrink call: fromDynArr failed" << std::endl;
}

int Matrix::convolve(const Matrix ker)
{
	if (reset() == nullptr)
		std::cerr << "Error in Matrix::convolve call: matrix does not exist" << std::endl;
	
	else if (ker.reset() == nullptr)
		std::cerr << "Error in Matrix::convolve call: kernel does not exist" << std::endl;
	
	else if (ker.row()%2==0 ||ker.col()%2==0)
		std::cerr << "Error in Matrix::convolve call: kernel dimensions must be odd values" << std::endl;

	else if (ker.row()>nRow || ker.col()>nCol)
		std::cerr << "Error in Matrix::convolve call: invalid kernel dimension(s)" << std::endl;
	
	else
	{
		Matrix res(nRow,nCol);	// matrice risultante
		
		const dimType shi = ker.row()/2;
		const dimType shj = ker.col()/2;
		
		for(dimType i=0; i<nRow; ++i)
		{
			for(dimType j=0; j<nCol; ++j)
			{
				for(dimType ki=0; ki<ker.row(); ++ki)
				{
					for(dimType kj=0; kj<ker.col(); ++kj)
					{
						res(i,j) += (*this)(i+ki-shi,j+kj-shj) * ker(ki,kj);
					}
				}
			}
		}
		
		*this = res;
		
		return EXIT_SUCCESS;
	}
	
	return EXIT_FAILURE;
}

inline int Matrix::convolve(const Matrix ker, const unsigned int iter)
{
	if (iter == 0)
		return EXIT_SUCCESS;
	
	for(unsigned int it = 0; it < iter; ++it)
	{
		if(Matrix::convolve(ker) == EXIT_FAILURE)
		{
			std::cerr << "Error in Matrix::convolve at iteration " << it+1 << ": unable to perform matrix convolution" << std::endl;
			return EXIT_FAILURE;
		}
	}
	
	return EXIT_SUCCESS;
}

#endif
