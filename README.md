# bpm-detector

`bpm-detector` calculates the beats per minute of an audio file.

## Building and running bpm-detector

To compile the project into a binary, call the following command in the working directory of the project:

$ make binary

This will create a binary file called `bpm-detector`. This program expects the relative path of a stereo audio file in the `wav` format; therefore, the audio file you want to test should be copied to the project directory. For instance, if you wanted to calculate the bpm for a file called `test.wav`, you would call the following:

$ ./bpm-detector test.wav

## How it works

The algorithm used to calculate the bpm is based largely on Eric D. Scheirer's paper, [Tempo and beat analysis of acoustic musical signals](http://www.iro.umontreal.ca/~pift6080/H09/documents/papers/scheirer_jasa.pdf), and the project called [Beat This](https://www.clear.rice.edu/elec301/Projects01/beat_sync/index.html) by Kileen Cheng, Bobak Nazer, Jyoti Uppuluri, and Ryan Verret.

The program reads in an audio file that is specified by the user as a command line argument. That audio file is then processed 5 seconds at a time. Every 5 seconds of data is broken into 6 frequency banded subbands. Each subband gets filtered to emphasize the energy peaks, or beats, of the signal. Then each subband is convolved with a comb filter that corresponds to a specific BPM under test. BPMs from 60 to 180 are tested. The convolution that outputs the greatest amount of energy is chosen as the winning bpm. The winning bpm values are tracked and summed for all subbands and all data chunks. At the end of program execution, the absolute winning bpm is printed.




