default: binary run

binary:
	g++ bpm_detector2.c kiss_fft.c kiss_fftr.c -o bpm_detector2

run:
	./bpm_detector2 fire.wav