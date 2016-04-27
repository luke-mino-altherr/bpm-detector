default: sequential omp mpi

sequential:
	mkdir -p bin/
	icc bpm_detector.c kiss_fft/kiss_fft.c kiss_fft/kiss_fftr.c -o bin/bpm-detector

omp:
	mkdir -p bin/
	icc -fopenmp -heap-arrays bpm_detector_open_mp.c kiss_fft/kiss_fft.c kiss_fft/kiss_fftr.c -o bin/bpm-detector-omp
	icc -fopenmp -heap-arrays bpm_detector_open_mp2.c kiss_fft/kiss_fft.c kiss_fft/kiss_fftr.c -o bin/bpm-detector-omp2

mpi:
	mkdir -p bin/
	mpiCC bpm_detector_mpi.c kiss_fft/kiss_fft.c kiss_fft/kiss_fftr.c -o bin/bpm-detector-mpi

clean:
	rm -r bin
	rm error* output*