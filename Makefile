default: binary run

binary:
	g++ bpmDetector.c BeatDetektor.cpp kiss_fftr.c kiss_fft.c wave.c -o bpmDetector

run:
	./bpmDetector