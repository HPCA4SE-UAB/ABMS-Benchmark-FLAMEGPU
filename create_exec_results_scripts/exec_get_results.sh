#!/bin/bash
# Process model execution data.
# Author: Juanjo Rodriguez, 01-05-2018

if [ "$#" -ne 2 ];
then
   echo "Usage: ./proces_exec.sh data_directory_name gpu_number(0, 1, ...)"
else
   directoryname=$1
   gpu=$2
   nexec=10
   totalexectime=0
   avgtime=0
   for ((i=0;i<$nexec;i++));
      do
         export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/home/juanjo/FLAMEGPU/glew/glew-2.1.0/lib:$LD_LIBRARY_PATH
         exectime=`./model $directoryname/0.xml 100 $gpu | grep "Total" | cut -d ' ' -f 4`
         totalexectime=`echo $totalexectime+$exectime | bc`
         echo "Execution "$((i+1))", time: "$exectime", total execution time: "$totalexectime
      done
   avgtime=`echo "scale=4; $totalexectime / 10" | bc -l`
   echo "Average time /10: "$avgtime
   avgtime=`echo "scale=4; $avgtime / 1000" | bc -l`
   echo "Average time /1000: "$avgtime
   printf "Average time: %.2f s.\n" $avgtime
fi

