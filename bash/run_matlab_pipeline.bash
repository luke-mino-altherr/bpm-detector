#!/bin/bash
#BSUB -L /bin/bash
#BSUB -J BpmDetector-Matlab-Pipelined
#BSUB -q ht-10g
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -n 12
work=/home/<insert_username>/bpm-detector/matlab/pipelined/
cd $work
matlab -logfile /home/<insert_username>/bpm-detector/matlab/logs/pipelined_1.txt -nodisplay -r "control('<RELATIVE_PATH_TO_AUDIO_FILE>', 1)" 
matlab -logfile /home/<insert_username>/bpm-detector/matlab/logs/pipelined_5.txt -nodisplay -r "control('<RELATIVE_PATH_TO_AUDIO_FILE>', 5)"
matlab -logfile /home/<insert_username>/bpm-detector/matlab/logs/pipelined_10.txt -nodisplay -r "control('<RELATIVE_PATH_TO_AUDIO_FILE>', 10)"
matlab -logfile /home/<insert_username>/bpm-detector/matlab/logs/pipelined_20.txt -nodisplay -r "control('<RELATIVE_PATH_TO_AUDIO_FILE>', 20)"
matlab -logfile /home/<insert_username>/bpm-detector/matlab/logs/pipelined_30.txt -nodisplay -r "control('<RELATIVE_PATH_TO_AUDIO_FILE>', 30)"
matlab -logfile /home/<insert_username>/bpm-detector/matlab/logs/pipelined_38.txt -nodisplay -r "control('<RELATIVE_PATH_TO_AUDIO_FILE>', 38)"
