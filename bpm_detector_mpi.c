//
// BPM Detector
//
// Created by Luke Mino-Altherr
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "kiss_fftr.h"
#include "bpm_detector.h"
#include "mpi.h"
#define TRUE 1
#define FALSE 0

unsigned char buffer4[4];
unsigned char buffer2[2];

FILE *ptr;

int main(int argc, char ** argv) {
    //Start timing code
    clock_t start_time, end_time;
    double cpu_time_used;
    int numprocs, rank, vect_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;

    MPI_Datatype WaveType;
    MPI_Datatype type[13] = { MPI_UNSIGNED_CHAR, MPI_UNSIGNED, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR,
                              MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED,
                              MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED_CHAR,
                              MPI_UNSIGNED};
    int blocklen[13] = { 4, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 4, 1};
    MPI_Aint disp[13];

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Start timing code
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = clock();

    // Get file path.
    if (argc != 2) /* argc should be 2 for correct execution */
    {
        printf("usage: %s relative filename", argv[0]);
        MPI_Finalize();
        return -1;
    }

    /***********************************************************************************/
    // Read in the header data to get the basic attributes of the file.
    /***********************************************************************************/
    WAVE *wave;
    wave = (WAVE *) malloc(sizeof(WAVE));

    disp[0] = (char*) &(wave->riff) - (char*) wave;
    disp[1] = (char*) &(wave->overall_size) - (char*) wave;
    disp[2] = (char*) &(wave->wave) - (char*) wave;
    disp[3] = (char*) &(wave->fmt_chunk_marker) - (char*) wave;
    disp[4] = (char*) &(wave->length_of_fmt) - (char*) wave;
    disp[5] = (char*) &(wave->format_type) - (char*) wave;
    disp[6] = (char*) &(wave->channels) - (char*) wave;
    disp[7] = (char*) &(wave->sample_rate) - (char*) wave;
    disp[8] = (char*) &(wave->byterate) - (char*) wave;
    disp[9] = (char*) &(wave->block_align) - (char*) wave;
    disp[10] = (char*) &(wave->bits_per_sample) - (char*) wave;
    disp[11] = (char*) &(wave->data_chunk_header) - (char*) wave;
    disp[12] = (char*) &(wave->data_size) - (char*) wave;
    MPI_Type_create_struct(13, blocklen, disp, type, &WaveType);
    MPI_Type_commit(&WaveType);

    int read;
    char *filename = NULL;
    filename = (char *) malloc(1024 * sizeof(char));
    
    if (getcwd(filename, 1024) != NULL) {
      strcat(filename, "/");
      strcat(filename, argv[1]);
    }


    if (rank == 0) {
        
        // open file
        ptr = fopen(filename, "r");
        if (ptr == NULL) {
            printf("Error opening file\n");
            return -1;
        }

        // Read in WAVE attributes
        read = fread(wave->riff, sizeof(wave->riff), 1, ptr);

        read = fread(buffer4, sizeof(buffer4), 1, ptr);

        // convert little endian to big endian 4 byte int
        wave->overall_size = buffer4[0] |
                             (buffer4[1] << 8) |
                             (buffer4[2] << 16) |
                             (buffer4[3] << 24);

        read = fread(wave->wave, sizeof(wave->wave), 1, ptr);

        read = fread(wave->fmt_chunk_marker, sizeof(wave->fmt_chunk_marker), 1, ptr);

        read = fread(buffer4, sizeof(buffer4), 1, ptr);

        // convert little endian to big endian 4 byte integer
        wave->length_of_fmt = buffer4[0] |
                              (buffer4[1] << 8) |
                              (buffer4[2] << 16) |
                              (buffer4[3] << 24);


        read = fread(buffer2, sizeof(buffer2), 1, ptr);

        wave->format_type = buffer2[0] | (buffer2[1] << 8);
        
	if (wave->format_type != 1) {
	    printf("Data must be in PCM format. Exiting. \n");
	    MPI_Finalize();
	    return -1;
	}

        read = fread(buffer2, sizeof(buffer2), 1, ptr);

        wave->channels = buffer2[0] | (buffer2[1] << 8);

        if (wave->channels != 2) {
            printf("Expected a stereo audio file. Exiting.\n");
	    MPI_Finalize();
            return -1;
        }

        read = fread(buffer4, sizeof(buffer4), 1, ptr);

        wave->sample_rate = buffer4[0] |
                            (buffer4[1] << 8) |
                            (buffer4[2] << 16) |
                            (buffer4[3] << 24);

        read = fread(buffer4, sizeof(buffer4), 1, ptr);

        wave->byterate = buffer4[0] |
                         (buffer4[1] << 8) |
                         (buffer4[2] << 16) |
                         (buffer4[3] << 24);

        read = fread(buffer2, sizeof(buffer2), 1, ptr);

        wave->block_align = buffer2[0] |
                            (buffer2[1] << 8);

        read = fread(buffer2, sizeof(buffer2), 1, ptr);

        wave->bits_per_sample = buffer2[0] |
                                (buffer2[1] << 8);

        read = fread(wave->data_chunk_header, sizeof(wave->data_chunk_header), 1, ptr);

        read = fread(buffer4, sizeof(buffer4), 1, ptr);

        wave->data_size = buffer4[0] |
                          (buffer4[1] << 8) |
                          (buffer4[2] << 16) |
                          (buffer4[3] << 24);

	
    }
    MPI_Bcast(wave, 1, WaveType, 0, MPI_COMM_WORLD);

    char format_name[10] = "";
    if (wave->format_type == 1)
        strcpy(format_name, "PCM");
    else if (wave->format_type == 6)
        strcpy(format_name, "A-law");
    else if (wave->format_type == 7)
        strcpy(format_name, "Mu-law");

    long size_of_each_sample = (wave->channels * wave->bits_per_sample) / 8;

    long num_samples = (wave->data_size) / (size_of_each_sample);

    // calculate duration of file
    float duration_in_seconds = (float) wave->overall_size / wave->byterate;

    int size_is_correct = TRUE;

    // make sure that the bytes-per-sample is completely divisible by num.of channels
    long bytes_in_each_channel = (size_of_each_sample / wave->channels);
    if ((bytes_in_each_channel * wave->channels) != size_of_each_sample) {
        printf("Error: %ld x %ud <> %ld\n", bytes_in_each_channel, wave->channels, size_of_each_sample);
        size_is_correct = FALSE;
    }

    if (!size_is_correct) {
        printf("Size is incorrect. Exiting.\n");
        MPI_Finalize();
        return -1;
    }

    // the valid amplitude range for values based on the bits per sample
    long low_limit = 0l;
    long high_limit = 0l;

    switch (wave->bits_per_sample) {
        case 8:
            low_limit = -128;
            high_limit = 127;
            break;
        case 16:
            low_limit = -32768;
            high_limit = 32767;
            break;
        case 32:
            low_limit = -2147483648;
            high_limit = 2147483647;
            break;
    }

    /***********************************************************************************/
    // Starting BPM Computation
    /***********************************************************************************/

    unsigned int i = 0, j = 0, k = 0;

    // Number of samples to analyze. We are taking 4 seconds of data.
    unsigned int N = 4 * wave->sample_rate;
    if (N % 2 != 0) N += 1;
    int loops = floor(num_samples / N);

    int minbpm = 60;
    int maxbpm = 180;
    int bpm_range = maxbpm - minbpm;
    int resolution = 1;
    int energy_size = bpm_range/resolution;
    float current_bpm = 0;
    int winning_bpm = 0;

    // Allocate subband buffers
    int num_sub_bands = 6;
    unsigned int sub_band_size = N / num_sub_bands;

    kiss_fft_scalar *sub_band_input;
    sub_band_input = (kiss_fft_scalar *) malloc(N*sizeof(kiss_fft_scalar));

    if (rank == 0) {
        /** Stage 0: Read in data from file **/

        // Allocate FFT buffers to read in data
        kiss_fft_scalar *fft_input_ch1, *fft_input_ch2;
        fft_input_ch1 = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
        fft_input_ch2 = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));

        // Data buffer to read from file into
        kiss_fft_scalar *data;
        data = (kiss_fft_scalar *) malloc(2 * sizeof(kiss_fft_scalar));

        // Temporarily hold data from file
        char *temp_data_buffer;
        temp_data_buffer = (char *) malloc(bytes_in_each_channel*sizeof(char));

        for (j = 0; j < loops; j++) {
            // Read in data from file to left and right channel buffers
            for (i = 0; i < N; i++) {
                // Loop for left and right channels
                for (k = 0; k < 2; k++) {
                    read = fread(temp_data_buffer, sizeof(temp_data_buffer), 1, ptr);

                    switch (bytes_in_each_channel) {
                        case 4:
                            data[k] = temp_data_buffer[0] | (temp_data_buffer[1] << 8) | (temp_data_buffer[2] << 16) | (temp_data_buffer[3] << 24);
                            break;
                        case 2:
                            data[k] = temp_data_buffer[0] |
                                      (temp_data_buffer[1] << 8);
                            break;
                        case 1:
                            data[k] = temp_data_buffer[0];
                            break;
                    }

                    // check if value was in range
                    if (data[k] < low_limit || data[k] > high_limit)
                        printf("**value out of range\n");
                }


                // Set data to FFT buffers
                fft_input_ch1[i] = (kiss_fft_scalar) data[0];
                fft_input_ch2[i] = (kiss_fft_scalar) data[1];
            } // End reading in data for left and right channels
            MPI_Send(fft_input_ch1, N, MPI_FLOAT, 1, 1, MPI_COMM_WORLD);
            MPI_Send(fft_input_ch2, N, MPI_FLOAT, 2, 1, MPI_COMM_WORLD);

        }
        free(fft_input_ch1);
        free(fft_input_ch2);
        free(temp_data_buffer);
        free(data);

        int *frequency_map;
        frequency_map = (int *) calloc(200, sizeof(int));

        MPI_Recv(frequency_map, 200, MPI_INT, 53, 8, MPI_COMM_WORLD, &status);

        winning_bpm = most_frequent_bpm(frequency_map);
        printf("BPM winner is: %i\n", winning_bpm);

        printf("\nDumping map...\n\n");
        dump_map(frequency_map);

        free(frequency_map);
    }
    else if (rank == 1 || rank == 2) {
        /** Stage 1: Filterbank **/

        kiss_fft_scalar *fft_input;
        fft_input = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));

        kiss_fft_scalar **sub_band_input_2d;
        sub_band_input_2d = (kiss_fft_scalar **) malloc(num_sub_bands * sizeof(kiss_fft_scalar *));
        for (i = 0; i < num_sub_bands; i++) {
            sub_band_input_2d[i] = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
        }

        for (j = 0; j < loops; j++) {
            MPI_Recv(fft_input, N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
            printf("Rank %s receiving %i\n",rank,j);
            sub_band_input_2d = filterbank(fft_input, sub_band_input_2d, N, wave->sample_rate);
            for (i = 0; i < num_sub_bands; i++)
                MPI_Send(sub_band_input_2d[i], N, MPI_FLOAT, (rank-1)*num_sub_bands+3+i, 2, MPI_COMM_WORLD);
        }

        free(fft_input);
        for (i = 0; i < num_sub_bands; i++) {
            free(sub_band_input_2d[i]);
        }
        free(sub_band_input_2d);
    }
    else if ( 2 < rank < 15 ) {
        /** Stage 2: Full-Wave_Rectification **/

        for (j = 0; j < loops; j++) {
            if (2 < rank < 9) MPI_Recv(sub_band_input, N, MPI_FLOAT, 1, 2, MPI_COMM_WORLD, &status);
            else MPI_Recv(sub_band_input, N, MPI_FLOAT, 2, 2, MPI_COMM_WORLD, &status);
            sub_band_input = full_wave_rectifier(sub_band_input, N);
            MPI_Send(sub_band_input, N, MPI_FLOAT, (rank+12), 3, MPI_COMM_WORLD);
        }
    }
    else if ( 14 < rank < 27 ) {
        /** Stage 3: Hanning Window Smoothing **/

        for (j = 0; j < loops; j++) {
            MPI_Recv(sub_band_input, N, MPI_FLOAT, (rank-12), 3, MPI_COMM_WORLD, &status);
            sub_band_input = hanning_window(sub_band_input, N, wave->sample_rate);
            MPI_Send(sub_band_input, N, MPI_FLOAT, (rank+12), 4, MPI_COMM_WORLD);
        }
    }
    else if ( 26 < rank < 39 ) {
        /** Stage 4: Differentiation and Half_wave Rectification **/

        for (j = 0; j < loops; j++) {
            MPI_Recv(sub_band_input, N, MPI_FLOAT, (rank-12), 4, MPI_COMM_WORLD, &status);
            sub_band_input = differentiator(sub_band_input, N);
            sub_band_input = half_wave_rectifier(sub_band_input, N);
            MPI_Send(sub_band_input, N, MPI_FLOAT, (rank+12), 5, MPI_COMM_WORLD);
        }
    }
    else if ( 38 < rank < 51 ) {
        /** Stage 5: Comb Filter Convolution **/

        double *energyA;
        energyA = (double *) malloc(sizeof(double)*energy_size);

        for (j = 0; j < loops; j++) {
            MPI_Recv(sub_band_input, N, MPI_FLOAT, (rank-12), 5, MPI_COMM_WORLD, &status);
            energyA = comb_filter_convolution(sub_band_input, energyA, N, wave->sample_rate, resolution, minbpm, maxbpm, high_limit);
            if ( 38 < rank < 45 ) MPI_Send(energyA, energy_size, MPI_DOUBLE, 51, 6, MPI_COMM_WORLD);
            else MPI_Send(energyA, energy_size, MPI_DOUBLE, 52, 6, MPI_COMM_WORLD);
        }

        free(energyA);
    }
    else if (rank == 51 || rank == 52) {
        /** Stage 6: Collect energy buffer for each subband **/

        double *energyB, *energyC;
        energyB = (double *) malloc(sizeof(double)*(energy_size));
        energyC = (double *) malloc(sizeof(double)*(energy_size));

        for (j = 0; j < loops; j++) {
            energyC = clear_energy_buffer(energyC, bpm_range, resolution);
            for (i = 0; i < num_sub_bands; i++) {
                if (rank == 51) MPI_Recv(energyB, (energy_size), MPI_DOUBLE, 39+i, 6, MPI_COMM_WORLD, &status);
                else MPI_Recv(energyB, (energy_size), MPI_DOUBLE, 45+i, 6, MPI_COMM_WORLD, &status);

                for (k = 0; k < (energy_size); k++ ) {
                    energyC[k] = energyC[k] + energyB[k];
                }
            }

            MPI_Send(energyC, energy_size, MPI_DOUBLE, 53, 7, MPI_COMM_WORLD);
        }

        free(energyB);
        free(energyC);
    }
    else if (rank == 53) {
        /** Stage 7: Compute bpm **/

        double *energy_ch1, *energy_ch2;
        energy_ch1 = (double *) malloc(sizeof(double)*(energy_size));
        energy_ch2 = (double *) malloc(sizeof(double)*(energy_size));
        int *frequency_map_1;
        frequency_map_1 = (int *) malloc(200 * sizeof(int));

        for (j = 0; j < loops; j++) {
            MPI_Recv(energy_ch1, energy_size, MPI_DOUBLE, 51, 7, MPI_COMM_WORLD, &status);
            MPI_Recv(energy_ch2, energy_size, MPI_DOUBLE, 52, 7, MPI_COMM_WORLD, &status);

            for (i = 0; i < energy_size; i++) {
                energy_ch1[i] = energy_ch1[i] + energy_ch2[i];
            }

            // Calculate the bpm from the total energy
            current_bpm = compute_bpm(energy_ch1, bpm_range, minbpm, resolution);

            if (current_bpm != -1) {
                printf("BPM computation is: %f\n", current_bpm);
                frequency_map_1[(int) round(current_bpm)] += 1;
            }

            printf("Current BPM winner is: %i\n", most_frequent_bpm(frequency_map_1));
        }

        MPI_Send(frequency_map_1, 200, MPI_INT, 0, 8, MPI_COMM_WORLD);

        free(frequency_map_1);
        free(energy_ch1);
        free(energy_ch2);
    }
    else if (rank == 54) {
      printf("(1-4): %s \n", wave->riff);
            //printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
            printf("(5-8) Overall size: bytes:%u, Kb:%u \n", wave->overall_size, wave->overall_size / 1024);
            printf("(9-12) Wave marker: %s\n", wave->wave);
            printf("(13-16) Fmt marker: %s\n", wave->fmt_chunk_marker);
            //printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
            printf("(17-20) Length of Fmt wave: %u \n", wave->length_of_fmt);
            //printf("%u %u \n", buffer2[0], buffer2[1]);
            printf("(21-22) Format type: %u %s \n", wave->format_type, format_name);
            //printf("%u %u \n", buffer2[0], buffer2[1]);
            printf("(23-24) Channels: %u \n", wave->channels);
            //printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
            printf("(25-28) Sample rate: %u\n", wave->sample_rate);
            //printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
            printf("(29-32) Byte Rate: %u , Bit Rate:%u\n", wave->byterate, wave->byterate * 8);
            //printf("%u %u \n", buffer2[0], buffer2[1]);
            printf("(33-34) Block Alignment: %u \n", wave->block_align);
            //printf("%u %u \n", buffer2[0], buffer2[1]);
            printf("(35-36) Bits per sample: %u \n", wave->bits_per_sample);
            printf("(37-40) Data Marker: %s \n", wave->data_chunk_header);
            //printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);
            printf("(41-44) Size of data chunk: %u \n", wave->data_size);
            printf("Size of each sample:%ld bytes\n", size_of_each_sample);
            printf("Number of samples:%lu \n", num_samples);
            printf("Approx.Duration in seconds=%f\n", duration_in_seconds);
            //printf("Approx.Duration in h:m:s=%s\n", seconds_to_time(duration_in_seconds));
            printf("\n.Valid range for data values : %ld to %ld \n", low_limit, high_limit);
            printf("loops is %i\n", loops);
    }

    if (ptr) {
        fclose(ptr);
        ptr = NULL;
    }

    free(wave);
    free(filename);

    // Finish timing program
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = clock();
    cpu_time_used = ((double)(end_time-start_time))/CLOCKS_PER_SEC;
    if (rank == 0) printf("\nProgram took %f seconds to complete.\n",cpu_time_used);

    MPI_Finalize();
    return 0;

}

kiss_fft_scalar ** filterbank(kiss_fft_scalar * time_data_in, kiss_fft_scalar ** filterbank,
                              unsigned int N, unsigned int sampling_rate){
    /*
     * Split time domain data in data_in into 6 separate frequency bands
     *   Band limits are [0 200 400 800 1600 3200]
     *
     * Output a vector of 6 time domain arrays of size N
     */
    // Initialize array of bandlimits
    int * bandlimits, *bandleft, *bandright;
    bandleft = (int *) malloc(sizeof(int) * 6);
    bandright = (int *) malloc(sizeof(int) * 6);
    bandlimits = (int *) malloc(sizeof(int) * 6);
    bandlimits[0] = 3;
    bandlimits[1] = 200;
    bandlimits[2] = 400;
    bandlimits[3] = 800;
    bandlimits[4] = 1600;
    bandlimits[5] = 3200;

    // Compute the boundaries of the bandlimits in terms of array location
    int i, j, maxfreq;
    maxfreq = sampling_rate/2;
    for (i = 0; i < 5; i++) {
        bandleft[i] = floor(bandlimits[i] * N/(2*maxfreq))+1;
        bandright[i] = floor(bandlimits[i+1] * N/(2*maxfreq));
    }
    bandleft[5] = floor(bandlimits[5]/maxfreq*N/2)+1;
    bandright[5] = floor(N/2);

    // Initialize FFT buffers
    kiss_fftr_cfg fft_cfg = kiss_fftr_alloc(N, 0, NULL, NULL);
    kiss_fftr_cfg fft_inv_cfg = kiss_fftr_alloc(N, 1, NULL, NULL);
    kiss_fft_cpx *freq_data_in, *freq_data_out;
    freq_data_in = (kiss_fft_cpx *) malloc(N*sizeof(kiss_fft_cpx));
    freq_data_out = (kiss_fft_cpx *) malloc(N*sizeof(kiss_fft_cpx));

    // Take FFT of input time domain data
    kiss_fftr(fft_cfg, time_data_in, freq_data_in);

    for (i = 0; i < 5; i++) {
        for (j = 0; j < N; j++) {
            freq_data_out[j].r = 0.0;
            freq_data_out[j].i = 0.0;
        }
        for (j = bandleft[i]; j < bandright[i]; j++) {
            freq_data_out[j].r = freq_data_in[j].r;
            freq_data_out[j].i = freq_data_in[j].i;
            freq_data_out[N-j].r = freq_data_in[N-j].r;
            freq_data_out[N-j].i = freq_data_in[N-j].i;
        }
        kiss_fftri(fft_inv_cfg, freq_data_out, filterbank[i]);
        /*for (j=0;j<N;j++) {
            printf("%f\n", filterbank[i][j]);
        }*/
        //printf("----------------------\n\n\n\n\n\n\n\n");
    }
    //sleep(10);
    free(bandlimits);
    free(bandleft);
    free(bandright);
    free(freq_data_in);
    free(freq_data_out);

    return filterbank;
}

kiss_fft_scalar * hanning_window(kiss_fft_scalar * data_in, unsigned int N, unsigned int sampling_rate) {
    /*
     * Implement a 200 ms half hanning window.
     *
     * Input is a signal in the frequency domain
     * Output is a windowed signal in the frequency domain.
     */
    kiss_fftr_cfg fft_window_cfg = kiss_fftr_alloc(N, 0, NULL, NULL);
    kiss_fftr_cfg fft_data_cfg = kiss_fftr_alloc(N, 0, NULL, NULL);
    kiss_fftr_cfg fft_data_inv_cfg = kiss_fftr_alloc(N, 1, NULL, NULL);
    kiss_fft_scalar *hanning_in;
    kiss_fft_cpx *hanning_out, *data_out;
    hanning_in = (kiss_fft_scalar *) malloc(N*sizeof(kiss_fft_scalar));
    hanning_out = (kiss_fft_cpx *) malloc(N*sizeof(kiss_fft_cpx));
    data_out = (kiss_fft_cpx *) malloc(N*sizeof(kiss_fft_cpx));

    int hann_len = .2*sampling_rate;

    int i;
    for (i = 0; i < N; i++) {
        if (i < hann_len)
            hanning_in[i] = pow(cos(2*i*M_PI/hann_len),2);
        else
            hanning_in[i] = 0.0;
        hanning_out[i].r = 0.0;
        hanning_out[i].i = 0.0;
    }

    kiss_fftr(fft_window_cfg, hanning_in, hanning_out);
    kiss_fftr(fft_data_cfg, data_in, data_out);

    for (i = 0; i < N; i++) {
        data_out[i].r *= hanning_out[i].r;
        data_out[i].i *= hanning_out[i].i;
    }

    kiss_fftri(fft_data_inv_cfg, data_out, data_in);

    free(hanning_in);
    free(hanning_out);
    free(data_out);
    kiss_fft_cleanup();

    return data_in;
}

kiss_fft_scalar * full_wave_rectifier(kiss_fft_scalar * input_buffer, unsigned int N) {
    /*
     * Rectifies a signal in the time domain.
     */
    int i;

    for (i = 1; i < N; i++) {
        if (input_buffer[i] < 0.0)
            input_buffer[i] *= -1.0;
    }

    return input_buffer;
}

kiss_fft_scalar * half_wave_rectifier(kiss_fft_scalar * input_buffer, unsigned int N) {
    /*
     * Rectifies a signal in the time domain.
     */
    int i;

    for (i = 1; i < N; i++) {
        if (input_buffer[i] < 0.0)
            input_buffer[i] = 0.0;
    }

    return input_buffer;
}

kiss_fft_scalar * differentiator(kiss_fft_scalar * input_buffer, unsigned int N) {
    /*
     * Differentiates a signal in the time domain.
     */
    kiss_fft_scalar * output;
    output = (kiss_fft_scalar *) malloc(N*sizeof(kiss_fft_scalar));

    int i;
    output[0] = 0;

    for (i = 1; i < N; i++) {
        output[i] = input_buffer[i] - input_buffer[i-1];
    }

    for (i = 0; i < N; i++) {
        input_buffer[i] = output[i];
    }

    free(output);

    return input_buffer;
}

double * comb_filter_convolution(kiss_fft_scalar * data_input, double * energy,
                               unsigned int N, unsigned int sample_rate, float resolution,
                               int minbpm, int maxbpm, int high_limit) {
    /*
     * Convolves the FFT of the data_input of size N with an impulse train
     * with a periodicity relative to the bpm in the range of minbpm to maxbpm.
     *
     * Returns energy array
     */
    kiss_fftr_cfg fft_cfg_filter = kiss_fftr_alloc(N, 0, NULL, NULL);
    kiss_fftr_cfg fft_cfg_data = kiss_fftr_alloc(N, 0, NULL, NULL);

    kiss_fft_scalar *filter_input, *filter_abs, *data_abs;
    filter_input = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_cpx));
    filter_abs = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_cpx));
    data_abs = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_cpx));

    kiss_fft_cpx *filter_output, *data_output;
    filter_output = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
    data_output = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));

    kiss_fftr(fft_cfg_data, data_input, data_output);
    data_abs = absolute_value(data_output, data_abs, N);

    int id;
    float i;
    unsigned int j, ti;
    unsigned int bpm_range = maxbpm - minbpm;

    for (i = 0; i < bpm_range; (i+=resolution)) {

        // Ti is the period of impulses (samples per beat)
        ti = (double)60/(minbpm + i)*sample_rate;
        //printf("ti: %i; bpm: %f\n", ti, (minbpm+i));

        for (j = 0; j < N; j++) {
            if (j%ti == 0) {
                filter_input[j] = (kiss_fft_scalar) high_limit;
                //printf("%f, %u, %u \n", i, j, ti);
            } else {
                filter_input[j] = 0.0;
            }
        }

        kiss_fftr(fft_cfg_filter, filter_input, filter_output);

        filter_abs = absolute_value(filter_output, filter_abs, N);

        id = i/resolution;

        for (j = 0; j < N/2; j++) {
            energy[id] += filter_abs[j] * data_abs[j];
        }

        //printf("Energy of bpm %f is %f\n", (minbpm+i), energy[id]);
    }

    free(filter_input);
    free(filter_output);
    free(filter_abs);
    free(data_output);
    free(data_abs);
    kiss_fft_cleanup();

    return energy;
}

float compute_bpm(double * energy_buffer, unsigned int bpm_range, unsigned int minbpm, float resolution){
    float bpm;
    bpm = (float)max_array(energy_buffer, bpm_range/resolution);
    if (bpm != -1) {
        bpm *= resolution;
        bpm += minbpm;
    }
    return bpm;

}

double * clear_energy_buffer(double * energy_buffer, unsigned int bpm_range, float resolution) {
    int i, id;
    for (i = 0; i < bpm_range; (i+=resolution)) {
        id = i/resolution;
        energy_buffer[id] = 0.0;
    }

    return energy_buffer;
}

kiss_fft_scalar * absolute_value(kiss_fft_cpx * input, kiss_fft_scalar * output, unsigned int N) {
    unsigned int i;

    for (i = 0; i < N; i++) {
        output[i] = pow(pow(input[i].r, 2) + pow(input[i].i ,2), .5);
    }

    return output;
}

int max_array(double * array, int size) {
    /*
     * Computes the max element of the array
     * and returns the corresponding index.
     */
    int i, index=0;
    double max = 0;
    for (i = 0; i < size; i++) {
        if (array[i] > max) {
            max = array[i];
            index = i;
        }
    }
    printf("Max value at %i is %f\n", index, max);
    if (max == 0.0) return -1;
    else return index;
}

int most_frequent_bpm(int * map) {
    int i, winner=0, value=0;
    for (i=0; i<200; i++) {
        if (map[i] > value) {
            winner = i;
            value = map[winner];
        }
    }
    return winner;
}

void dump_map(int * map) {
    int i;
    for (i = 0; i < 200; i++) {
        printf("BPM: %i, Count: %i\n", i, map[i]);
    }

}

