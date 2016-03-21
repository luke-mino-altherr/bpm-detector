default: binary run

binary:
	mkdir -p bin/
	g++ bpm_detector.c kiss_fft.c kiss_fftr.c -o bin/bpm-detector

omp:
	mkdir -p bin/
	g++ bpm_detector_open_mp.c kiss_fft.c kiss_fftr.c -fopenmp -o bin/bpm-detector-omp

run:
	./bin/bpm-detector fire.wav

clean:
	rm -r bin
	rm error* output*