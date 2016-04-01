#!/bin/sh
#BSUB -J BpmDetector-MPI
#BSUB -o output_file_mpi
#BSUB -e error_file_mpi
#BSUB -n 10
#BSUB -q ht-10g
#BSUB cwd /home/mino-altherr.l/bpmDetector

work=/home/mino-altherr.l/bpmDetector
cd $work
tempfile1=hostlistrun
tempfile2=hostlist-tcp
echo $LSB_MCPU_HOSTS > $tempfile1
declare -a hosts
read -a hosts < ${tempfile1}
for ((i=0; i<${#hosts[@]}; i += 2)) ;
  do
   HOST=${hosts[$i]}
   CORE=${hosts[(($i+1))]}
   echo $HOST:$CORE >> $tempfile2
done

#mpirun -np 60 valgrind --tool=memcheck --log-file=output bin/bpm-detector-mpi Samples/fire.wav
mpirun -np 60 -prot -TCP -lsf ./bin/bpm-detector-mpi Samples/fire.wav

