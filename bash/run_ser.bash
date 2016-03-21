#!/bin/sh
#BSUB -J BpmDetector-Serial
#BSUB -o output_file_ser
#BSUB -e error_file_ser
#BSUB -n 1
#BSUB -q ht-10g
#BSUB cwd /home/<INSERT USERNAME>/bpm-detector

work=/home/<INSERT USERNAME>/bpm-detector
cd $work
./bin/bpm-detector <INSERT RELATIVE PATH TO AUDIO FILE>