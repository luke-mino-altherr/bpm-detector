#!/bin/sh
#BSUB -J BpmDetector-OMP
#BSUB -o output_file_omp
#BSUB -e error_file_omp
#BSUB -n 10
#BSUB -q ht-10g
#BSUB cwd /home/mino-altherr.l/bpmDetector

work=/home/mino-altherr.l/bpmDetector
cd $work
export OMP_NUM_THREADS=32
./bpm-detector-omp fire.wav