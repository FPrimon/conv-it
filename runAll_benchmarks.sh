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
INPUT=data/input
KERNEL=data/kernel
OUTPUT=data/output

iters=(10 50 250)
rowKer=(55 15 3)
colKer=(55 15 3)
m=${#iters[*]}

rI=1000
cI=1000

inFile=$INPUT/input-$rI-$cI.txt

for (( j=0 ; j<m ; j++ ))
do
	iter=${iters[j]}
	rK=${rowKer[j]}
	cK=${colKer[j]}
	
	kerFile=$KERNEL/kernel-$rK-$cK.txt
	outFile=$OUTPUT/$label-out-input-$rI-$cI-kernel-$rK-$cK.txt
	
	echo "Input matrix: $rI x $cI" >> $logFile
	echo "Kernel matrix: $rK x $cK" >> $logFile
    echo "Number of iterations: $iter" >> $logFile
	((logTot++))
	echo -n "."
	
	$launchName $iter $rI $cI $rK $cK $inFile $kerFile $outFile >> $logFile 2>&1	# fa il merge di std::cerr in std::cout
	
	status=$?
	
	echo "Exit status: $status" >> $logFile
	echo "" >> $logFile
	
	if (( status==0 ));		# QUI !!
	then
		echo "Input matrix: $rI x $cI" >> $diffFile
		echo "Kernel matrix: $rK x $cK" >> $diffFile
        echo "Number of iterations: $iter" >> $diffFile
		
		result="$OUTPUT/final-it-$iter-i-$rI-$cI-k-$rK-$cK.txt"
		diff -q $outFile $result >> $diffFile	# -q (--brief) dice solo se i due file differiscono, -s fa il viceversa
		
		((diffFail+=$?))
		((logSucc++))
		
		echo "" >> $diffFile
	fi
done
