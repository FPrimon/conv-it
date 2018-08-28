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
#ifndef MATRIX_CONV_MPI
#define MATRIX_CONV_MPI

#include <cstdlib>
#include <iostream>
#include <mpi.h>

#include "conv_it_def.hpp"
#include "matrixConv.hpp"

//	CLASSes DECLARATIONs
class ChunkMPI;
class MatrixMPI;

//	FUNCTIONs DECLARATIONs

//	CLASSes DEFINITIONs
class ChunkMPI
{
	//	MEMBERs DEFINITIONs
	private:
		int worldRank=-1;
		int worldSize=0;
		
		MPI_Comm headComm = MPI_COMM_NULL;
		int headRank=-1;
		int headSize=0;
		
		MPI_Comm lineComm = MPI_COMM_NULL;
		int lineRank=-1;
		int lineSize=0;
		
		bool doubleHead = false;
		bool firstPal = false;
		bool lastPal = false;
		
		dimType height=0;
	
	//	CONSTRUCTORs DECLARATIONs
	public:
		ChunkMPI() = default;
		ChunkMPI(const dimType matRow, const Matrix ker);
		
	//	MEMBER-FUNCTIONs DEFINITIONs
	public:
		int wRank() const	{ return worldRank; }
		int wSize() const	{ return worldSize; }
		
		int lRank() const	{ return lineRank; }
		int lSize() const	{ return lineSize; }
		
		dimType h() const	{ return height; }
		
		void check() const
		{
			std::cout << "WORLD \t Size: " << worldSize << " \t Rank: " << worldRank << std::endl;
			std::cout << "HEAD \t Size: " << headSize << " \t Rank: " << headRank << std::endl;
			std::cout << "LINE \t Size: " << lineSize << " \t Rank: " << lineRank << std::endl;
			std::cout << "PAL \t First: " << firstPal << " \t Last: " << lastPal << std::endl;
			std::cout << "Height: " << height << std::endl;
		}
	
	//	MEMBER-FUNCTIONs DECLARATIONs
	public:
		int sendRecvRows(MatrixMPI &mat, const dimType shi) const;
		int spreadData(MatrixMPI &mat, Matrix ker) const;
		int shareData(MatrixMPI &mat, const dimType shi) const;
		int collectData(MatrixMPI &mat, const dimType nRow) const;
		
	//	DESTRUCTOR
	public:
		~ChunkMPI() = default;
};

class MatrixMPI : virtual public Matrix	// Pointer-like Class
{
	// MEMBERs DEFINITIONs
	protected:
		ChunkMPI proc;
	
	// CONSTRUCTORs
	public:
		MatrixMPI() = default;		// costruisce la matrice 0x0
		MatrixMPI(const dimType nR, const dimType nC, const ChunkMPI p) : Matrix(nR,nC), proc(p)	{}
		MatrixMPI( const dimType nR, const dimType nC, const Matrix ker) : proc(nR,ker)
		{
			MatrixMPI tmp(proc.h(),nC,proc);
			*this = tmp;
		}
	
	//	COPY-CONSTRUCTOR
	public:
		MatrixMPI(const MatrixMPI &mat) : Matrix(mat), proc(mat.proc)	{}	// copia gli attributi, ma non alloca ulteriore memoria
		
	//	OPERATORs
	public:
		MatrixMPI& operator=(const MatrixMPI &rightMat)		// copia gli attributi, ma non alloca ulteriore memoria
		{
			Matrix::operator=(rightMat);
			proc=rightMat.proc;
			return *this;
		}
	
	// MEMBER-FUNCTIONs DEFINITIONs
	public:
		void check() const	{ proc.check(); }
	
	// MEMBER-FUNCTIONs DECLARATIONs
	protected:
		int convolve(const Matrix ker)	override;
	public:
		int convolve(const Matrix ker, const unsigned int iter)	override;
			
	//	DESTRUCTOR
	public:
		~MatrixMPI() = default;
};

//	FUNCTIONs DEFINTIONs
inline void MPIinitializer(int *argc, char ***argv)
{
	int MPIFlag=0;
	MPI_Initialized(&MPIFlag);
	
	if (MPIFlag==0)
		MPI_Init(argc,argv);
}

//	CONSTRUCTORs DEFINITIONs
inline ChunkMPI::ChunkMPI(dimType matRow, Matrix ker)
{
	MPIinitializer(nullptr,nullptr);
	
	// world
	MPI_Comm_rank(MPI_COMM_WORLD,&worldRank);
	MPI_Comm_size(MPI_COMM_WORLD,&worldSize);
	
	int lines;
	
	for(lines=2; lines*ker.row() <= matRow; ++lines);	// quante righe di processi riesco a fare?
	
	if (worldSize <= (--lines)*static_cast<signedDimType>(ker.cell()))
	{
		int splitKey = MPI_UNDEFINED;
		
		// head
		if (worldRank < lines)
			splitKey=1;
		
		MPI_Comm_split(MPI_COMM_WORLD,splitKey,worldRank,&headComm);
		if (headComm != MPI_COMM_NULL)
		{
			MPI_Comm_rank(headComm,&headRank);
			MPI_Comm_size(headComm,&headSize);
			
			height = matRow/headSize;
			int residue = matRow%headSize;
			
			if (residue != 0)
			{	
				doubleHead = true;
			
				if (headRank < residue)
					++height;
			
				splitKey=height;
				
				MPI_Comm tmp;
				MPI_Comm_dup(headComm,&tmp);
				
				MPI_Comm_split(tmp,splitKey,worldRank,&headComm);
				MPI_Comm_rank(headComm,&headRank);
				MPI_Comm_size(headComm,&headSize);
				
				MPI_Comm_free(&tmp);
			}
		}
		
		// line
		splitKey = worldRank%lines;
		MPI_Comm_split(MPI_COMM_WORLD,splitKey,worldRank,&lineComm);
		
		MPI_Comm_rank(lineComm,&lineRank);
		MPI_Comm_size(lineComm,&lineSize);
		
		// pals
		if (worldSize > lines && worldSize%lines!=0 )	// worldSize > lines
		{
			if (worldRank == lines-1)
				firstPal = true;
			if (worldRank == worldSize-lines)
				lastPal = true;
		}
		else
		{
			if (worldRank == worldSize-1)
				firstPal = true;
			if (worldRank == 0)
				lastPal = true;
		}
		
		// height
		MPI_Bcast(&height,1,MAT_DIMTYPE,0,lineComm);
	}
	else if (worldRank == 0)
		std::cerr << "Error in ChunkMPI constructor call: too many processes" << std::endl;
}

//	MEMBER-FUNCTIONs DEFINITIONs
inline int ChunkMPI::sendRecvRows(MatrixMPI &mat, const dimType shi) const	// worldSize > 1
{
	const dimType buffSize = shi*mat.col();
	elemType *const above = new elemType[buffSize]();
	elemType *const below = new elemType[buffSize]();
	
	int count=0;
	MPI_Status status;
	
	if (worldRank == 0)
	{
		// mando le righe sotto
		MPI_Send(mat.reset()+mat.cell()-buffSize,buffSize,MAT_ELEMTYPE,1,21,MPI_COMM_WORLD);
		// ricevo le righe sotto
		MPI_Recv(below,buffSize,MAT_ELEMTYPE,1,23,MPI_COMM_WORLD,&status);
		
		MPI_Get_count(&status,MAT_ELEMTYPE,&count);
		if (count != static_cast<signedDimType>(buffSize))
		{
			std::cerr << "Error in ChunkMPI::spreadData call: MPI_Recv failed \t Process rank: 0" << std::endl;
			delete [] above;
			delete [] below;
			return EXIT_FAILURE;
		}
		
		/*if(lastPal)
		{
			MPI_Sendrecv(mat.reset(),buffSize,MAT_ELEMTYPE,worldSize-1,29,above,buffSize,MAT_ELEMTYPE,worldSize-1,29,MPI_COMM_WORLD,&status);
		}
		else
			MPI_Recv(above,buffSize,MAT_ELEMTYPE,MPI_ANY_SOURCE,25,MPI_COMM_WORLD,&status);
		
		MPI_Get_count(&status,MAT_ELEMTYPE,&count);
		if (count != buffSize)
		{
			std::cerr << "Error in ChunkMPI::spreadData call: MPI_Recv failed \t Process rank: 0" << std::endl;
			delete [] above;
			delete [] below;
			return EXIT_FAILURE;
		}*/
	}
	else if (worldRank == worldSize-1)
	{
		// ricevo le righe sopra
		MPI_Recv(above,buffSize,MAT_ELEMTYPE,worldRank-1,21,MPI_COMM_WORLD,&status);
		
		MPI_Get_count(&status,MAT_ELEMTYPE,&count);
		if (count != static_cast<signedDimType>(buffSize))
		{
			std::cerr << "Error in ChunkMPI::spreadData call: MPI_Recv failed \t Process rank: " << worldRank << std::endl;
			delete [] above;
			delete [] below;
			return EXIT_FAILURE;
		}
		// mando le righe sopra
		MPI_Send(mat.reset(),buffSize,MAT_ELEMTYPE,worldRank-1,23,MPI_COMM_WORLD);
		
		/*if(firstPal)
		{
			MPI_Sendrecv(mat.reset()+mat.cell()-buffSize,buffSize,MAT_ELEMTYPE,0,29,below,buffSize,MAT_ELEMTYPE,0,29,MPI_COMM_WORLD,&status);
		}
		else
			MPI_Recv(below,buffSize,MAT_ELEMTYPE,MPI_ANY_SOURCE,27,MPI_COMM_WORLD,&status);
			
		MPI_Get_count(&status,MAT_ELEMTYPE,&count);
		if (count != buffSize)
		{
			std::cerr << "Error in ChunkMPI::spreadData call: MPI_Recv failed \t Process rank: " << worldRank << std::endl;
			delete [] above;
			delete [] below;
			return EXIT_FAILURE;
		}*/
	}
	else
	{
		// mando le righe sotto, ricevo le righe sopra
		MPI_Sendrecv(mat.reset()+mat.cell()-buffSize,buffSize,MAT_ELEMTYPE,worldRank+1,21,above,buffSize,MAT_ELEMTYPE,worldRank-1,21,MPI_COMM_WORLD,&status);
		
		MPI_Get_count(&status,MAT_ELEMTYPE,&count);
		if (count != static_cast<signedDimType>(buffSize))
		{
			std::cerr << "Error in ChunkMPI::spreadData call: MPI_Recv failed \t Process rank: " << worldRank << std::endl;
			delete [] above;
			delete [] below;
			return EXIT_FAILURE;
		}
		// mando le righe sopra, ricevo le righe sotto
		MPI_Sendrecv(mat.reset(),buffSize,MAT_ELEMTYPE,worldRank-1,23,below,buffSize,MAT_ELEMTYPE,worldRank+1,23,MPI_COMM_WORLD,&status);
		
		MPI_Get_count(&status,MAT_ELEMTYPE,&count);
		if (count != static_cast<signedDimType>(buffSize))
		{
			std::cerr << "Error in ChunkMPI::spreadData call: MPI_Recv failed \t Process rank: " << worldRank << std::endl;
			delete [] above;
			delete [] below;
			return EXIT_FAILURE;
		}
		
		/*if(firstPal)
			MPI_Send(mat.reset()+mat.cell()-buffSize,buffSize,MAT_ELEMTYPE,0,25,MPI_COMM_WORLD);
			
		if(lastPal)
			MPI_Send(mat.reset(),buffSize,MAT_ELEMTYPE,worldSize-1,27,MPI_COMM_WORLD);*/
	}
	
	if (worldRank == 0)
	{
		MPI_Recv(above,buffSize,MAT_ELEMTYPE,MPI_ANY_SOURCE,25,MPI_COMM_WORLD,&status);
		
		MPI_Get_count(&status,MAT_ELEMTYPE,&count);
		if (count != static_cast<signedDimType>(buffSize))
		{
			std::cerr << "Error in ChunkMPI::spreadData call: MPI_Recv failed \t Process rank: 0" << std::endl;
			delete [] above;
			delete [] below;
			return EXIT_FAILURE;
		}
	}
	else if(firstPal)
		MPI_Send(mat.reset()+mat.cell()-buffSize,buffSize,MAT_ELEMTYPE,0,25,MPI_COMM_WORLD);
	
	if (worldRank == worldSize-1)
	{
		MPI_Recv(below,buffSize,MAT_ELEMTYPE,MPI_ANY_SOURCE,27,MPI_COMM_WORLD,&status);
			
		MPI_Get_count(&status,MAT_ELEMTYPE,&count);
		if (count != static_cast<signedDimType>(buffSize))
		{
			std::cerr << "Error in ChunkMPI::spreadData call: MPI_Recv failed \t Process rank: " << worldRank << std::endl;
			delete [] above;
			delete [] below;
			return EXIT_FAILURE;
		}
	}	
	else if(lastPal)
		MPI_Send(mat.reset(),buffSize,MAT_ELEMTYPE,worldSize-1,27,MPI_COMM_WORLD);
	
	mat.addRows(shi,above,below);
	
	delete [] above;
	delete [] below;
	
	return EXIT_SUCCESS;
}

inline int ChunkMPI::spreadData(MatrixMPI &mat, Matrix ker) const	// modifico mat alla fine
{
	if (lineComm == MPI_COMM_NULL)
	{
		if (worldRank == 0)
			std::cerr << "Error in ChunkMPI::spreadData call: uninitialized chunk" << std::endl;
		return EXIT_FAILURE;
	}
	
	elemType *buffer = nullptr;		// NO ALIASING
	
	if (doubleHead && headRank==0)
	{
		const dimType nCell = mat.col()*height*headSize;	// numero di celle della propria parte
		
		if (worldRank == 0)
		{
			MPI_Send(mat.reset()+nCell,mat.cell()-nCell,MAT_ELEMTYPE,headSize,21,MPI_COMM_WORLD);
			
			if (mat.fromDynArr(height*headSize,mat.col(),mat.reset()) == EXIT_FAILURE)
			{
				std::cerr << "Error in ChunkMPI::spreadData call: mat.fromDynArr failed \t Process rank: 0" << std::endl;
				return EXIT_FAILURE;
			}
		}
		else
		{
			buffer = new elemType[nCell]();
			int count=0;
			MPI_Status status;
			
			MPI_Recv(buffer,nCell,MAT_ELEMTYPE,0,21,MPI_COMM_WORLD,&status);
			MPI_Get_count(&status,MAT_ELEMTYPE,&count);
			
			if (count != static_cast<signedDimType>(nCell))
			{
				std::cerr << "Error in ChunkMPI::spreadData call: MPI_Recv failed \t Process rank: " << worldRank << std::endl;
				delete [] buffer;
				return EXIT_FAILURE;
			}
		}
	}
	
	if (worldRank == 0)
	{
		mat.toDynArr(buffer);
		const MatrixMPI tmp(height,mat.col(),*this);
		mat = tmp;
	}
	
	// ogni testa riceve il suo pezzo di matrice...
	if (headComm != MPI_COMM_NULL)
		MPI_Scatter(buffer,mat.cell(),MAT_ELEMTYPE,mat.reset(),mat.cell(),MAT_ELEMTYPE,0,headComm);
	
	// ... e lo manda a tutti i membri della stessa riga
	MPI_Bcast(mat.reset(),mat.cell(),MAT_ELEMTYPE,0,lineComm);
	
	// mando il ker a tutti
	MPI_Bcast(ker.reset(),ker.cell(),MAT_ELEMTYPE,0,MPI_COMM_WORLD);
	
	if (buffer!=nullptr)
		delete [] buffer;
	
	const dimType shi = ker.row()/2;
	
	if (shi > 0)
		return sendRecvRows(mat,shi);
	else
		return EXIT_SUCCESS;
}

inline int ChunkMPI::shareData(MatrixMPI &mat, const dimType shi) const
{
	if (lineComm == MPI_COMM_NULL)
	{
		if (worldRank == 0)
			std::cerr << "Error in ChunkMPI::shareData call: uninitialized chunk" << std::endl;
		return EXIT_FAILURE;
	}
	
	/*if (shi > 0)
		mat.rwShrink(shi);*/
	
	if (lineSize > 1)
	{
		elemType *const buffer = new elemType[mat.cell()]();
		MPI_Allreduce(mat.reset(),buffer,mat.cell(),MAT_ELEMTYPE,MPI_SUM,lineComm);
		
		if(mat.fromDynArr(mat.row(),mat.col(),buffer) == EXIT_FAILURE)
		{
			std::cerr << "Error in ChunkMPI::shareData call: mat.fromDynArr failed \t Process rank: " << worldRank << std::endl;
			delete [] buffer;
			return EXIT_FAILURE;
		}
		
		delete [] buffer;
	}
	
	if (shi > 0)
		return sendRecvRows(mat,shi);
	else
		return EXIT_SUCCESS;
}

inline int ChunkMPI::collectData(MatrixMPI &mat, const dimType nRow) const
{
	if (lineComm == MPI_COMM_NULL)
	{
		if (worldRank == 0)
			std::cerr << "Error in ChunkMPI::collectData call: uninitialized chunk" << std::endl;
		return EXIT_FAILURE;
	}
	
	/*if (shi > 0)
		mat.rwShrink(shi);*/
	
	elemType *buffer=nullptr;
	
	if (lineRank == 0)
		buffer = new elemType[mat.cell()]();
	
	MPI_Reduce(mat.reset(),buffer,mat.cell(),MAT_ELEMTYPE,MPI_SUM,0,lineComm);
	
	if (lineRank == 0)
	{
		if(mat.fromDynArr(mat.row(),mat.col(),buffer) == EXIT_FAILURE)
		{
			std::cerr << "Error in ChunkMPI::spreadData call: mat.fromDynArr failed \t Process rank: " << worldRank << std::endl;
			delete [] buffer;
			return EXIT_FAILURE;
		}
		
		delete [] buffer;
		buffer = nullptr;
	}
	
	if (worldRank == 0)
		buffer = new elemType[nRow*mat.col()]();
	else if (headRank == 0)
		buffer = new elemType[headSize*mat.cell()]();
	
	if (headComm != MPI_COMM_NULL)
		MPI_Gather(mat.reset(),mat.cell(),MAT_ELEMTYPE,buffer,mat.cell(),MAT_ELEMTYPE,0,headComm);
	
	if (doubleHead && headRank==0)
	{
		const dimType nCell = headSize*mat.cell();	// numero di celle della propria parte
		
		if (worldRank == 0)
		{
			int count=0;
			MPI_Status status;
			
			MPI_Recv(buffer+nCell,nRow*mat.col()-nCell,MAT_ELEMTYPE,headSize,23,MPI_COMM_WORLD,&status);
			MPI_Get_count(&status,MAT_ELEMTYPE,&count);
			
			if (count != static_cast<signedDimType>(nRow*mat.col()-nCell))
			{
				std::cerr << "Error in ChunkMPI::collectData call: MPI_Recv failed \t Process rank: 0" << std::endl;
				return EXIT_FAILURE;
			}
		}
		else
		{
			MPI_Send(buffer,nCell,MAT_ELEMTYPE,0,23,MPI_COMM_WORLD);
		}
	}
	
	if (worldRank == 0 && mat.fromDynArr(nRow,mat.col(),buffer) == EXIT_FAILURE)
	{
		std::cerr << "Error in ChunkMPI::spreadData call: mat.fromDynArr failed \t Process rank: 0" << std::endl;
		delete [] buffer;
		return EXIT_FAILURE;
	}
	
	if (buffer != nullptr)
		delete [] buffer;
	
	return EXIT_SUCCESS;
}

inline int MatrixMPI::convolve(const Matrix ker)
{
	if (reset() == nullptr)
		std::cerr << "Error in MatrixMPI::convolve call: matrix does not exist" << std::endl;
	
	else if (ker.reset() == nullptr)
		std::cerr << "Error in MatrixMPI::convolve call: kernel does not exist" << std::endl;
	
	else if (ker.row()%2==0 ||ker.col()%2==0)
		std::cerr << "Error in MatrixMPI::convolve call: kernel dimensions must be odd values" << std::endl;

	else if (ker.row()>nRow || ker.col()>nCol)
		std::cerr << "Error in MatrixMPI::convolve call: invalid kernel dimension(s)" << std::endl;
	
	else
	{
		const dimType kCol=ker.col();
		const dimType shj = kCol/2;
		
		const dimType kFirst = ker.cell()*proc.lRank()/proc.lSize();
		const dimType kLast = ker.cell()*(proc.lRank()+1)/proc.lSize();
		
		dimType ki=0;
		dimType kj=0;
		
		MatrixMPI res(nRow-ker.row()+1,nCol,proc);	// matrice risultante
		
		const elemType *const idxKer = ker.reset();
		
		for(dimType i=0; i<res.nRow; ++i)
		{
			for(dimType j=0; j<res.nCol; ++j)
			{
				for(dimType kKer=kFirst; kKer < kLast; ++kKer)
				{
					ki = kKer/kCol;
					kj = kKer%kCol;
					
					res(i,j) += (*this)(i+ki,j+kj-shj) * idxKer[kKer];
				}
			}
		}
		
		*this = res;
		
		return EXIT_SUCCESS;
	}
	
	return EXIT_FAILURE;
}

inline int MatrixMPI::convolve(const Matrix ker, const unsigned int iter)
{
	if (iter == 0)
		return EXIT_SUCCESS;
	
	if (proc.wSize() == 1)
		return Matrix::convolve(ker,iter);
	
	const dimType nR = nRow;
	const dimType shi = ker.row()/2;
	
	if (proc.spreadData(*this,ker) == EXIT_FAILURE)
	{
		if (proc.wRank() == 0)
			std::cerr << "Error in MatrixMPI::convolve call: unable to properly deliver data" << std::endl;
		return EXIT_FAILURE;
	}
	
	if(MatrixMPI::convolve(ker) == EXIT_FAILURE)
	{
		if (proc.wRank() == 0)
			std::cerr << "Error in MatrixMPI::convolve at iteration " << 1 << ": unable to perform matrix convolution" << std::endl;
		return EXIT_FAILURE;
	}
	
	for(unsigned int it = 1; it < iter; ++it)
	{
		if (proc.shareData(*this,shi) == EXIT_FAILURE)
		{
			if (proc.wRank() == 0)
				std::cerr << "Error in MatrixMPI::convolve call: unable to properly share data" << std::endl;
			return EXIT_FAILURE;
		}
		
		if(MatrixMPI::convolve(ker) == EXIT_FAILURE)
		{
			if (proc.wRank() == 0)
				std::cerr << "Error in MatrixMPI::convolve at iteration " << it+1 << ": unable to perform matrix convolution" << std::endl;
			return EXIT_FAILURE;
		}
	}
	
	return proc.collectData(*this,nR);
}

#endif
