##
 #  This file is part of 'Conv-It' (https://github.com/FPrimon/conv-it).
 #  'Conv-It' is a program for iterative matrix convolution.
 #  Copyright © 2018 Francesco Primon.
 #
 #  'Conv-It' is free software: you can redistribute it and/or modify
 #  it under the terms of the GNU General Public License as published by
 #  the Free Software Foundation, either version 3 of the License, or
 #  any later version.
 #
 #  'Conv-It' is distributed in the hope that it will be useful,
 #  but WITHOUT ANY WARRANTY; without even the implied warranty of
 #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 #  GNU General Public License for more details.
 #
 #  You should have received a copy of the GNU General Public License
 #  along with 'Conv-It'. If not, see <https://www.gnu.org/licenses/>.
 ##
### SETTINGS ###
# choose one or more versions LABELS=(st omp mpi hybrid cuda)
LABELS=(omp mpi cuda)
# number of threads, multiple options allowed - OMPTHREADS=(2 4 6 8)
OMPTHREADS=(6)
# number of mpi processes, multiple options allowed - MPIPROCS=(2 4 8)
MPIPROCS=(4)

mkdir -p log

for label in ${LABELS[*]}
do
	echo -n "Dealing with $label version"
	exeName=conv_it_$label
	logFile=log/conv_it_$label.log
	diffFile=log/conv_it_$label.res
	
	logSucc=0
	logTot=0
	diffFail=0
	
	echo "This is the $exeName log file" > $logFile
	echo "" >> $logFile
	
	echo "This is the $exeName diff file" > $diffFile
	echo "" >> $diffFile
	
	STARTtime=$SECONDS
	
	case $label in
		( st )
			launchName=./$exeName
			
			echo "Launching $exeName ..." >> $logFile
			echo "Launching diff..." >> $diffFile
			
			echo "" >> $logFile
			echo "" >> $diffFile
			
			source runAll_benchmarks.sh;;
		( omp )
			launchName=./$exeName
			
			for nt in ${OMPTHREADS[*]}
			do
				export OMP_NUM_THREADS=$nt
				
				echo "Launching $exeName ..." >> $logFile
				echo "Launching diff..." >> $diffFile
				
				echo "OMP_NUM_THREADS = $OMP_NUM_THREADS" >> $logFile
				echo "OMP_NUM_THREADS = $OMP_NUM_THREADS" >> $diffFile
				
				echo "" >> $logFile
				echo "" >> $diffFile
				
				source runAll_benchmarks.sh
			done;;
		( mpi )
			for np in ${MPIPROCS[*]}
			do
				launchName="mpirun -np $np $exeName"
				
				echo "Launching $exeName ..." >> $logFile
				echo "Launching diff..." >> $diffFile
				
				echo "MPI_WORLD_SIZE = $np" >> $logFile
				echo "MPI_WORLD_SIZE = $np" >> $diffFile
				
				echo "" >> $logFile
				echo "" >> $diffFile
				
				source runAll_benchmarks.sh
			done;;
		( hybrid )
			for np in ${MPIPROCS[*]}
			do
				launchName="mpirun -np $np $exeName"
				
				for nt in ${OMPTHREADS[*]}
				do
					export OMP_NUM_THREADS=$nt
					
					if (( np*nt < 10 ));
					then
						echo "Launching $exeName ..." >> $logFile
						echo "Launching diff..." >> $diffFile
				
						echo "MPI_WORLD_SIZE = $np" >> $logFile
						echo "MPI_WORLD_SIZE = $np" >> $diffFile
					
						echo "OMP_NUM_THREADS = $OMP_NUM_THREADS" >> $logFile
						echo "OMP_NUM_THREADS = $OMP_NUM_THREADS" >> $diffFile
					
						echo "" >> $logFile
						echo "" >> $diffFile
				
						source runAll_benchmarks.sh
					fi
				done
			done;;
		( cuda )
			launchName="/usr/bin/time -f %e ./$exeName"
			
			echo "Launching $exeName ..." >> $logFile
			echo "Launching diff..." >> $diffFile
			
			echo "" >> $logFile
			echo "" >> $diffFile
			
			source runAll_benchmarks.sh;;
		( * )
			echo "Error: invalid label" >> $logFile
			echo "" >> $logFile
	esac
	
	ENDtime=$SECONDS
	ELAPSED=$(( $ENDtime - $STARTtime ))
	
	echo "Execution time: $ELAPSED s" >> $logFile
	echo "Successful launch(es): $logSucc out of $logTot" >> $logFile

	echo "Incorrect result(s): $diffFail out of $logSucc" >> $diffFile
	echo "DONE"
done
