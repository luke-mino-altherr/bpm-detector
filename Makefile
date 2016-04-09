default: sequential omp mpi

sequential:
	mkdir -p bin/
	g++ bpm_detector.c kiss_fft/kiss_fft.c kiss_fft/kiss_fftr.c -o bin/bpm-detector

omp:
	mkdir -p bin/
	g++ bpm_detector_open_mp.c kiss_fft/kiss_fft.c kiss_fft/kiss_fftr.c -fopenmp -o bin/bpm-detector-omp

mpi:
	mkdir -p bin/
	mpiCC bpm_detector_mpi.c kiss_fft/kiss_fft.c kiss_fft/kiss_fftr.c -o bin/bpm-detector-mpi

run-sequential:
	./bpm-detector samples/fire.wav

clean:
	rm -r bin
	rm error* output*