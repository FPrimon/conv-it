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
#ifndef CONV_IT_DEF
#define CONV_IT_DEF

typedef long int signedDimType;
typedef unsigned long int dimType;
typedef long int elemType;

#define MAT_DIMTYPE MPI_UNSIGNED_LONG
#define MAT_ELEMTYPE MPI_LONG

//	FUNCTIONs DEFINITIONs
inline dimType getIndex(signedDimType i, signedDimType j, const signedDimType nRow, const signedDimType nCol)
{
	//signedDimType nRow = nR;
	//signedDimType nCol = nC;
	
	if(nRow<=0 || nCol<=0)
	{
		std::cerr << "Error in getIndex call: non-positive matrix dimension(s)" << std::endl;
		return 0;
	}
	
	for( ; i<0 ; i+=nRow);
	for( ; j<0 ; j+=nCol);

	for( ; i>=nRow ; i-=nRow);
	for( ; j>=nCol ; j-=nCol);
	
	return static_cast<dimType> (i*nCol+j);
}

#endif
