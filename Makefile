default: binary run

binary:
	g++ bpm_detector.c kiss_fft.c kiss_fftr.c -o bpm-detector

run:
	./bpm-detector some_chords.wav