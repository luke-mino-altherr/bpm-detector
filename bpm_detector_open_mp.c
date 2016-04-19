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
#include "kiss_fft/kiss_fftr.h"
#include "bpm_detector.h"
#include <omp.h>
#define TRUE 1
#define FALSE 0

unsigned char buffer4[4];
unsigned char buffer2[2];

FILE *ptr;

char *temp_data_buffer;
double *energy;
kiss_fft_scalar *data;
kiss_fft_scalar **sub_band_input_ch1, **sub_band_input_ch2;
kiss_fft_scalar *fft_input_ch1, *fft_input_ch2;

#pragma omp threadprivate(temp_data_buffer, energy, data, sub_band_input_ch1, sub_band_input_ch2, fft_input_ch1, fft_input_ch2)

int main(int argc, char ** argv) {
    //Start timing code
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    // Get file path.
    if ( argc < 2 ) /* argc should be 2 for correct execution */
    {
        printf( "usage: %s relative filename\n", argv[0] );
        return -1;
    }

    char * filename = NULL;
    filename = (char *) malloc(1024 * sizeof(char));

    if (getcwd(filename, 1024) != NULL) {
        strcat(filename, "/");
        strcat(filename, argv[1]);
    }

    // open file
    printf("Opening file %s...\n", filename);
    ptr = fopen(filename, "r");
    if (ptr == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    /***********************************************************************************/
    // Read in the header data to get the basic attributes of the file.
    /***********************************************************************************/

    int read = 0;
    WAVE * wave;
    wave = (WAVE *) malloc(sizeof(WAVE));

    // Read in WAVE attributes
    read = fread(wave->riff, sizeof(wave->riff), 1, ptr);
    printf("(1-4): %s \n", wave->riff);

    read = fread(buffer4, sizeof(buffer4), 1, ptr);
    //printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

    // convert little endian to big endian 4 byte int
    wave->overall_size = buffer4[0] |
                         (buffer4[1]<<8) |
                         (buffer4[2]<<16) |
                         (buffer4[3]<<24);

    printf("(5-8) Overall size: bytes:%u, Kb:%u \n", wave->overall_size, wave->overall_size/1024);

    read = fread(wave->wave, sizeof(wave->wave), 1, ptr);
    printf("(9-12) Wave marker: %s\n", wave->wave);

    read = fread(wave->fmt_chunk_marker, sizeof(wave->fmt_chunk_marker), 1, ptr);
    printf("(13-16) Fmt marker: %s\n", wave->fmt_chunk_marker);

    read = fread(buffer4, sizeof(buffer4), 1, ptr);
    //printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

    // convert little endian to big endian 4 byte integer
    wave->length_of_fmt = buffer4[0] |
                          (buffer4[1] << 8) |
                          (buffer4[2] << 16) |
                          (buffer4[3] << 24);
    printf("(17-20) Length of Fmt wave: %u \n", wave->length_of_fmt);

    read = fread(buffer2, sizeof(buffer2), 1, ptr);
    //printf("%u %u \n", buffer2[0], buffer2[1]);

    wave->format_type = buffer2[0] | (buffer2[1] << 8);
    char format_name[10] = "";
    if (wave->format_type == 1)
        strcpy(format_name,"PCM");
    else if (wave->format_type == 6)
        strcpy(format_name, "A-law");
    else if (wave->format_type == 7)
        strcpy(format_name, "Mu-law");

    printf("(21-22) Format type: %u %s \n", wave->format_type, format_name);

    read = fread(buffer2, sizeof(buffer2), 1, ptr);
    //printf("%u %u \n", buffer2[0], buffer2[1]);

    wave->channels = buffer2[0] | (buffer2[1] << 8);
    printf("(23-24) Channels: %u \n", wave->channels);
    if (wave->channels != 2) {
        printf("Assumed there would be two channels. Exiting.");
        exit(1);
    }

    read = fread(buffer4, sizeof(buffer4), 1, ptr);
    //printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

    wave->sample_rate = buffer4[0] |
                        (buffer4[1] << 8) |
                        (buffer4[2] << 16) |
                        (buffer4[3] << 24);

    printf("(25-28) Sample rate: %u\n", wave->sample_rate);

    read = fread(buffer4, sizeof(buffer4), 1, ptr);
    //printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

    wave->byterate  = buffer4[0] |
                      (buffer4[1] << 8) |
                      (buffer4[2] << 16) |
                      (buffer4[3] << 24);
    printf("(29-32) Byte Rate: %u , Bit Rate:%u\n", wave->byterate, wave->byterate*8);

    read = fread(buffer2, sizeof(buffer2), 1, ptr);
    //printf("%u %u \n", buffer2[0], buffer2[1]);

    wave->block_align = buffer2[0] |
                        (buffer2[1] << 8);
    printf("(33-34) Block Alignment: %u \n", wave->block_align);

    read = fread(buffer2, sizeof(buffer2), 1, ptr);
    //printf("%u %u \n", buffer2[0], buffer2[1]);

    wave->bits_per_sample = buffer2[0] |
                            (buffer2[1] << 8);
    printf("(35-36) Bits per sample: %u \n", wave->bits_per_sample);

    read = fread(wave->data_chunk_header, sizeof(wave->data_chunk_header), 1, ptr);
    printf("(37-40) Data Marker: %s \n", wave->data_chunk_header);

    read = fread(buffer4, sizeof(buffer4), 1, ptr);
    //printf("%u %u %u %u\n", buffer4[0], buffer4[1], buffer4[2], buffer4[3]);

    wave->data_size = buffer4[0] |
                      (buffer4[1] << 8) |
                      (buffer4[2] << 16) |
                      (buffer4[3] << 24 );
    printf("(41-44) Size of data chunk: %u \n", wave->data_size);

    long size_of_each_sample = (wave->channels * wave->bits_per_sample) / 8;
    printf("Size of each sample:%ld bytes\n", size_of_each_sample);

    long num_samples = (wave->data_size) / (size_of_each_sample);
    printf("Number of samples:%lu \n", num_samples);

    // calculate duration of file
    float duration_in_seconds = (float) wave->overall_size / wave->byterate;
    printf("Approx.Duration in seconds=%f\n", duration_in_seconds);
    //printf("Approx.Duration in h:m:s=%s\n", seconds_to_time(duration_in_seconds));

    // read each sample from data chunk if PCM
    if (wave->format_type != 1) { // PCM
        printf("expected a PCM wav file type. Exiting.");
        return -1;
    }

    int  size_is_correct = TRUE;

    // make sure that the bytes-per-sample is completely divisible by num.of channels
    long bytes_in_each_channel = (size_of_each_sample / wave->channels);
    if ((bytes_in_each_channel  * wave->channels) != size_of_each_sample) {
        printf("Error: %ld x %ud <> %ld\n", bytes_in_each_channel, wave->channels, size_of_each_sample);
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

    printf("\n.Valid range for data values : %ld to %ld \n", low_limit, high_limit);

    /***********************************************************************************/
    // Start reading audio data and computing bpm
    /***********************************************************************************/

    unsigned int i = 0, j = 0, k = 0;

    // Number of samples to analyze. We are taking 4 seconds of data.
    unsigned int N = 4 * wave->sample_rate;
    if (N % 2 != 0) N += 1;
    int loops;
    if (arc > 2) loops = argv[2];
    else loops = floor(num_samples / N);
    printf("loops is %i\n", loops);

    int minbpm = 60;
    int maxbpm = 180;
    int bpm_range = maxbpm - minbpm;
    int resolution = 1;
    float current_bpm = 0;
    int winning_bpm = 0;

    int *frequency_map;
    frequency_map = (int *) calloc(200, sizeof(int));

    int num_sub_bands = 6;
    unsigned int sub_band_size = N / num_sub_bands;

    omp_set_num_threads(loops/2);

#pragma omp parallel private(i, k, current_bpm)
    {

#pragma omp critical
        {
            // Temporarily hold data from file
            temp_data_buffer = (char *) malloc(bytes_in_each_channel * sizeof(char));

            // Allocate history buffer to hold energy values for each tested BPM
            energy = (double *) malloc(sizeof(double) * (bpm_range / resolution));

            // Data buffer to read from file into
            data = (kiss_fft_scalar *) malloc(2 * sizeof(kiss_fft_scalar));

            // Allocate FFT buffers to read in data
            fft_input_ch1 = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
            fft_input_ch2 = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));

            // Allocate subband buffers
            sub_band_input_ch1 = (kiss_fft_scalar **) malloc(num_sub_bands * sizeof(kiss_fft_scalar *));
            sub_band_input_ch2 = (kiss_fft_scalar **) malloc(num_sub_bands * sizeof(kiss_fft_scalar *));
            for (i = 0; i < num_sub_bands; i++) {
                sub_band_input_ch1[i] = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
                sub_band_input_ch2[i] = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
                for (k = 0; k < N; k++) {
                    sub_band_input_ch1[i][k] = 0.0;
                    sub_band_input_ch2[i][k] = 0.0;
                }
            }
        }

#pragma omp for
        for (j = 0; j < loops; j++) {
            // Read in data from file to left and right channel buffers
            for (i = 0; i < N; i++) {
                // Loop for left and right channels
                for (k = 0; k < 2; k++) {
                    read = fread(temp_data_buffer, sizeof(temp_data_buffer), 1, ptr);

                    switch (bytes_in_each_channel) {
                        case 4:
                            data[k] = temp_data_buffer[0] |
                                      (temp_data_buffer[1] << 8) |
                                      (temp_data_buffer[2] << 16) |
                                      (temp_data_buffer[3] << 24);
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
                } // End reading in data for left and right channels

                // Set data to FFT buffers
                fft_input_ch1[i] = (kiss_fft_scalar) data[0];
                fft_input_ch2[i] = (kiss_fft_scalar) data[1];
            } // End read in data

            // Split data into separate frequency bands
            sub_band_input_ch1 = filterbank(fft_input_ch1, sub_band_input_ch1, N, wave->sample_rate);
            sub_band_input_ch2 = filterbank(fft_input_ch2, sub_band_input_ch2, N, wave->sample_rate);

            // Clear energy buffer before calculating new energy data
            energy = clear_energy_buffer(energy, bpm_range, resolution);

            // Filter and sum energy for each tested bpm for each subband
            for (i = 0; i < num_sub_bands; i++) {
                // Channel 1
                sub_band_input_ch1[i] = full_wave_rectifier(sub_band_input_ch1[i], N);
                sub_band_input_ch1[i] = hanning_window(sub_band_input_ch1[i], N, wave->sample_rate);
                sub_band_input_ch1[i] = differentiator(sub_band_input_ch1[i], N);
                sub_band_input_ch1[i] = half_wave_rectifier(sub_band_input_ch1[i], N);
                energy = comb_filter_convolution(sub_band_input_ch1[i], energy, N, wave->sample_rate,
                                                 1, minbpm, maxbpm, high_limit);

                // Channel 2
                sub_band_input_ch2[i] = full_wave_rectifier(sub_band_input_ch2[i], N);
                sub_band_input_ch2[i] = hanning_window(sub_band_input_ch2[i], N, wave->sample_rate);
                sub_band_input_ch2[i] = differentiator(sub_band_input_ch2[i], N);
                sub_band_input_ch2[i] = half_wave_rectifier(sub_band_input_ch2[i], N);
                energy = comb_filter_convolution(sub_band_input_ch2[i], energy, N, wave->sample_rate,
                                                 1, minbpm, maxbpm, high_limit);
            }

            // Calculate the bpm from the total energy
            current_bpm = compute_bpm(energy, bpm_range, minbpm, resolution);

            if (current_bpm != -1) {
                printf("BPM computation is: %f\n", current_bpm);
#pragma omp atomic
                frequency_map[(int) round(current_bpm)] += 1;
            }

            printf("Current BPM winner is: %i\n", most_frequent_bpm(frequency_map));

        }

        for (i = 0; i < num_sub_bands; i++) {
            free(sub_band_input_ch1[i]);
            free(sub_band_input_ch2[i]);
        }
        free(sub_band_input_ch1);
        free(sub_band_input_ch2);
        free(fft_input_ch1);
        free(fft_input_ch2);
        free(temp_data_buffer);
        free(data);
    }

    winning_bpm = most_frequent_bpm(frequency_map);
    printf("BPM winner is: %i\n", winning_bpm);

    printf("\nDumping map...\n\n");
    dump_map(frequency_map);

    if (ptr) {
        printf("Closing file..\n");
        fclose(ptr);
        ptr = NULL;
    }

    free(wave);
    free(filename);

    // Finish timing program
    end = clock();
    cpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("\nProgram took %f seconds to complete.\n",cpu_time_used);

}

kiss_fft_scalar ** filterbank(kiss_fft_scalar * time_data_in, kiss_fft_scalar ** filterbank,
                              unsigned int N, unsigned int sampling_rate){
    /*
     * Split time domain data in data_in into 6 separate frequency bands
     *   Band limits are [0 200 400 800 1600 3200]
     *
     * Output a vector of 6 time domain arrays of size N
     */
    // Initialize FFT buffers and array for the filterbank
    kiss_fft_cpx *freq_data_in, *freq_data_out;
    int * bandlimits, *bandleft, *bandright;
    kiss_fftr_cfg fft_cfg, fft_inv_cfg;

    #pragma omp critical
    {
        fft_cfg = kiss_fftr_alloc(N, 0, NULL, NULL);
        fft_inv_cfg = kiss_fftr_alloc(N, 1, NULL, NULL);
        freq_data_in = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
        freq_data_out = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));

        bandleft = (int *) malloc(sizeof(int) * 6);
        bandright = (int *) malloc(sizeof(int) * 6);
        bandlimits = (int *) malloc(sizeof(int) * 6);
    }

    // Initialize array of bandlimits
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


    // Take FFT of input time domain data
    kiss_fftr(fft_cfg, time_data_in, freq_data_in);

    for (i = 0; i < 6; i++) {
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
    kiss_fft_scalar *hanning_in;
    kiss_fft_cpx *hanning_out, *data_out, *temp_data;
    kiss_fftr_cfg fft_window_cfg, fft_data_cfg, fft_data_inv_cfg;

    #pragma omp critical
    {
        hanning_in = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
        hanning_out = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
        data_out = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
        temp_data = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));

        fft_window_cfg = kiss_fftr_alloc(N, 0, NULL, NULL);
        fft_data_cfg = kiss_fftr_alloc(N, 0, NULL, NULL);
        fft_data_inv_cfg = kiss_fftr_alloc(N, 1, NULL, NULL);
    }

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
    hanning_in[0] = 0.0;

    kiss_fftr(fft_window_cfg, hanning_in, hanning_out);
    kiss_fftr(fft_data_cfg, data_in, data_out);

    for (i = 0; i < N; i++) {
        temp_data[i].r = data_out[i].r * hanning_out[i].r - data_out[i].i * hanning_out[i].i;
        temp_data[i].i = data_out[i].i * hanning_out[i].r + data_out[i].r * hanning_out[i].i;
    }

    kiss_fftri(fft_data_inv_cfg, data_out, data_in);

    free(hanning_in);
    free(hanning_out);
    free(data_out);
    free(temp_data);
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

    #pragma omp critical
    {
        output = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
    }
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
    kiss_fft_scalar *filter_input;
    kiss_fft_cpx *filter_output, *data_output;
    kiss_fftr_cfg fft_cfg_filter, fft_cfg_data;

    #pragma omp critical
    {
        filter_input = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_cpx));
        filter_output = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
        data_output = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
        fft_cfg_filter = kiss_fftr_alloc(N, 0, NULL, NULL);
        fft_cfg_data = kiss_fftr_alloc(N, 0, NULL, NULL);
    }

    kiss_fftr(fft_cfg_data, data_input, data_output);
    data_abs = absolute_value(data_output, data_abs, N);

    int id;
    float i, a;
    unsigned int j, ti;
    double temp_energy_r, temp_energy_i;
    unsigned int bpm_range = maxbpm - minbpm;

    for (i = 0; i < bpm_range; (i+=resolution)) {

        // Ti is the period of impulses (samples per beat)
        ti = (double)60/(minbpm + i)*sample_rate;

        for (j = 0; j < N; j++) {
            if (j%ti == 0) {
                filter_input[j] = (kiss_fft_scalar) high_limit;
            } else {
                filter_input[j] = 0.0;
            }
        }

        kiss_fftr(fft_cfg_filter, filter_input, filter_output);

        id = i/resolution;

        for (j = 0; j < N/2; j++) {
            a = pow(.5, j/ti);
            a *= (60+.1*i)/maxbpm;
            temp_energy_r = (filter_output[j].r * data_output[j].r - filter_output[j].i * data_output[j].i);
            temp_energy_i = (filter_output[j].r * data_output[j].i + filter_output[j].i * data_output[j].r);

            energy[id] += pow(pow(temp_energy_i, 2) + pow(temp_energy_r, 2), .5)*a;
        }

        //printf("Energy of bpm %f is %f\n", (minbpm+i), energy[id]);
    }

    free(filter_input);
    free(filter_output);
    free(data_output);
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
