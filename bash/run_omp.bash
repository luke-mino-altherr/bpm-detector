#!/bin/sh
#BSUB -J BpmDetector-OMP
#BSUB -o output_file_omp
#BSUB -e error_file_omp
#BSUB -n 20
#BSUB -q ht-10g
#BSUB cwd /home/<INSERT USERNAME>/bpmDetector

work=/home/<INSERT USERNAME>/bpmDetector
cd $work
./bin/bpm-detector-omp <INSERT RELATIVE PATH TO AUDIO FILE>
