default: binary run

binary:
	g++ bpm_detector.c kiss_fft.c -o bpm_detector

run:
	./bpm_detector