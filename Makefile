default: binary run

binary:
	g++ bpm_detector.c kiss_fft.c kiss_fftr.c -o bpm_detector

run:
	./bpm_detector2 fire.wav