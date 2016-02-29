# bpm-detector

`bpm-detector` calculates the beats per minute of an audio file.

## Building and running bpm-detector

To compile the project into a binary, call the following command in the working directory of the project:

    $ make binary

This will create a binary file called `bpm-detector`. This program expects the relative path of a stereo audio file in the `wav` format as a command line argument; therefore, the audio file you want to test should be copied to the project directory. For instance, if you wanted to calculate the bpm for a file called `test.wav`, you would call the following:

    $ ./bpm-detector test.wav

## How it works

The algorithm used to calculate the bpm is based largely on Eric D. Scheirer's paper, [Tempo and beat analysis of acoustic musical signals](http://www.iro.umontreal.ca/~pift6080/H09/documents/papers/scheirer_jasa.pdf), and the project called [Beat This](https://www.clear.rice.edu/elec301/Projects01/beat_sync/index.html) by Kileen Cheng, Bobak Nazer, Jyoti Uppuluri, and Ryan Verret.

The program reads in an audio file that is specified by the user as a command line argument. That audio file is then processed 5 seconds at a time. Every 5 seconds of data is broken into 6 frequency banded subbands. Each subband gets filtered to emphasize the energy peaks, or beats, of the signal. Then each subband is convolved with a comb filter that corresponds to a specific BPM under test. BPMs from 60 to 180 are tested. The bpm corresponding to the convolution that outputs the greatest amount of energy is chosen as the winning bpm. The winning bpm values are tracked and summed for all subbands and all data chunks. At the end of program execution, the absolute winning bpm is printed.

### Frequency Filterbank

The chunk of audio data is processed into two channels: left and right audio channels. Each channel is then bandpassed and broken into 6 frequency bands and analyzed separately. This is important because usually different frequency bands will output different rhythmic pulses. For instance, in dance music, we are typically concerned with the lowest frequency band where the simple rhythmic bass drum lives. By filtering data into frequency bands we are getting additional meaningful data.

### Hanning Window

Each signal band is first full-wave rectified and then convolved with a 200-ms half hanning window. This filter smoothes the amplitude envelope of the signal, similar to the auditory system in humans.

### Differentiator

For each signal band, the first order difference is calculated to emphasize changes in amplitude. The signal is then half-wave rectified to remove imperfections in the filtered signal.

### Comb Filter

Each filtered signal band is then convolved with various comb filters. Each comb filter is an impulse train with a periodicity that corresponds to the BPM under test. A comb filter is generated for each BPM in the range 60 to 120. Each comb filter is then convolved with the the signal band data. The BPM that generate the greatest amount of energy in the convolution is returned as the winning bpm for that signal band. The BPMs are stored and summed across all signal bands and all time chunks. At the end of the program the absolute winning BPM is printed.



