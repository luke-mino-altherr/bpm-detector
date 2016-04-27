#!/bin/sh
#BSUB -J BpmDetector-MPI
#BSUB -o output_file_mpi
#BSUB -e error_file_mpi
#BSUB -n 56
#BSUB -q ser-par-10g
#BSUB cwd /home/<insert_username>/bpm-detector

work=/home/<insert_username>/bpm-detector
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

mpirun -np 56 -prot -TCP -lsf ./bin/bpm-detector-mpi <relative_path_to_audio_file> 1
mpirun -np 56 -prot -TCP -lsf ./bin/bpm-detector-mpi <relative_path_to_audio_file> 5
mpirun -np 56 -prot -TCP -lsf ./bin/bpm-detector-mpi <relative_path_to_audio_file> 10
mpirun -np 56 -prot -TCP -lsf ./bin/bpm-detector-mpi <relative_path_to_audio_file> 20
mpirun -np 56 -prot -TCP -lsf ./bin/bpm-detector-mpi <relative_path_to_audio_file> 30
mpirun -np 56 -prot -TCP -lsf ./bin/bpm-detector-mpi <relative_path_to_audio_file> 