#!/bin/sh
#BSUB -J BpmDetector-Serial
#BSUB -o output_file_ser
#BSUB -e error_file_ser
#BSUB -n 1
#BSUB -q ht-10g
#BSUB cwd /home/<insert_username>/bpm-detector

work=/home/<insert_username>/bpm-detector
cd $work
./bin/bpm-detector <relative_path_to_audio_file> 1
./bin/bpm-detector <relative_path_to_audio_file> 5
./bin/bpm-detector <relative_path_to_audio_file> 10
./bin/bpm-detector <relative_path_to_audio_file> 20
./bin/bpm-detector <relative_path_to_audio_file> 30
./bin/bpm-detector <relative_path_to_audio_file> 
