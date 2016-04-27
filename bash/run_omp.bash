#!/bin/sh
#BSUB -J BpmDetector-OMP
#BSUB -o output_file_omp
#BSUB -e error_file_omp
#BSUB -n 32
#BSUB -q ser-par-10g
#BSUB cwd /home/<insert_username>/bpm-detector

export OMP_NUM_THREADS=32

work=/home/<insert_username>/bpm-detector
cd $work

./bin/bpm-detector-omp <relative_path_to_audio_file> 1
./bin/bpm-detector-omp <relative_path_to_audio_file> 5
./bin/bpm-detector-omp <relative_path_to_audio_file> 10
./bin/bpm-detector-omp <relative_path_to_audio_file> 20
./bin/bpm-detector-omp <relative_path_to_audio_file> 30
./bin/bpm-detector-omp <relative_path_to_audio_file>

./bin/bpm-detector-omp2 <relative_path_to_audio_file> 1
./bin/bpm-detector-omp2 <relative_path_to_audio_file> 5
./bin/bpm-detector-omp2 <relative_path_to_audio_file> 10
./bin/bpm-detector-omp2 <relative_path_to_audio_file> 20
./bin/bpm-detector-omp2 <relative_path_to_audio_file> 30
./bin/bpm-detector-omp2 <relative_path_to_audio_file>
