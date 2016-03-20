default: binary run

binary:
	g++ bpm_detector.c kiss_fft.c kiss_fftr.c -o bpm-detector

omp:
	g++ bpm_detector_open_mp.c fopenmp -o bpm-detector-omp

run:
	./bpm-detector fire.wav