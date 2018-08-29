#LABELS=(st omp mpi hybrid cuda)
LABELS=(omp mpi)
#OMPTHREADS=(1 2 3 4)
OMPTHREADS=(6)
#MPIPROCS=(1 2 3 4)
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
