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
#define TRUE 1
#define FALSE 0

unsigned char buffer4[4];
unsigned char buffer2[2];

FILE *ptr;

int main(int argc, char ** argv) {
    //Start timing code
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    // Get file path.
    if ( argc != 2 ) /* argc should be 2 for correct execution */
    {
        printf( "usage: %s relative filename", argv[0] );
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
    if (wave->format_type == 1) { // PCM

        int  size_is_correct = TRUE;

        // make sure that the bytes-per-sample is completely divisible by num.of channels
        long bytes_in_each_channel = (size_of_each_sample / wave->channels);
        if ((bytes_in_each_channel  * wave->channels) != size_of_each_sample) {
            printf("Error: %ld x %ud <> %ld\n", bytes_in_each_channel, wave->channels, size_of_each_sample);
            size_is_correct = FALSE;
        }

        if (size_is_correct) {
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

            // Number of samples to analyze. We are taking 5 seconds of data.
            unsigned int N = 4 * wave->sample_rate;
            if (N % 2 != 0) N += 1;
            int loops = floor(num_samples / N);

            // Temporarily hold data from file
            char *temp_data_buffer;
            temp_data_buffer = (char *) calloc(bytes_in_each_channel, sizeof(char));

            // Used to recursively call comb filter bpm detection
            float a, b, c, d;

            int minbpm = 60;
            int maxbpm = 180;
            int bpm_range = maxbpm - minbpm;
            float current_bpm = 0;
            int winning_bpm = 0;

            int *frequency_map;
            frequency_map = (int *) calloc(200, sizeof(int));

            // Allocate history buffer to hold energy values for each tested BPM
            double *energy_history;
            double energy;
            energy_history = (double *) calloc(bpm_range, sizeof(double));

            // Data buffer to read from file into
            kiss_fft_scalar *data;
            data = (kiss_fft_scalar *) calloc(2, sizeof(kiss_fft_scalar));

            // Allocate FFT buffers to read in data
            kiss_fft_scalar *fft_input_ch1, *fft_input_ch2;
            fft_input_ch1 = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
            fft_input_ch2 = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));

            // Split data of both channels into 6 different sub_bands
            int num_sub_bands = 6;
            unsigned int sub_band_size = N / num_sub_bands;
            kiss_fft_scalar **sub_band_input_ch1, **sub_band_input_ch2, **sub_band_output_ch1, **sub_band_output_ch2;
            sub_band_input_ch1 = (kiss_fft_scalar **) malloc(num_sub_bands * sizeof(kiss_fft_scalar *));
            sub_band_input_ch2 = (kiss_fft_scalar **) malloc(num_sub_bands * sizeof(kiss_fft_scalar *));
            sub_band_output_ch1 = (kiss_fft_scalar **) malloc(num_sub_bands * sizeof(kiss_fft_scalar *));
            sub_band_output_ch2 = (kiss_fft_scalar **) malloc(num_sub_bands * sizeof(kiss_fft_scalar *));
            for (i = 0; i < num_sub_bands; i++) {
                sub_band_input_ch1[i] = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
                sub_band_input_ch2[i] = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
                sub_band_output_ch1[i] = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
                sub_band_output_ch2[i] = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
                for (k = 0; k < N; k++) {
                    sub_band_input_ch1[i][k] = 0.0;
                    sub_band_input_ch2[i][k] = 0.0;
                    sub_band_output_ch1[i][k] = 0.0;
                    sub_band_output_ch2[i][k] = 0.0;
                }
            }


            // Read in data from file to left and right channel buffers
            for (j = 0; j < loops; j++) {
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
                }

                // Populate sub-bands
                for (i = 0; i < num_sub_bands; i++) {
                    sub_band_input_ch1[i] = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
                    sub_band_input_ch2[i] = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
                    sub_band_output_ch1[i] = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
                    sub_band_output_ch2[i] = (kiss_fft_scalar *) malloc(N * sizeof(kiss_fft_scalar));
                    for (k = 0; k < N; k++) {
                        sub_band_input_ch1[i][k] = 0.0;
                        sub_band_input_ch2[i][k] = 0.0;
                        sub_band_output_ch1[i][k] = 0.0;
                        sub_band_output_ch2[i][k] = 0.0;
                    }
                    for (k = i * sub_band_size; k < (i + 1) * sub_band_size; k++) {
                        sub_band_input_ch1[i][k] = fft_input_ch1[k];
                        sub_band_input_ch2[i][k] = fft_input_ch2[k];
                    }
                }

                // Filter and compute bpm
                for (i = 0; i < num_sub_bands; i++) {
                    // Channel 1
                    //sub_band_input_ch1[i] = hanning_window(sub_band_input_ch1[i], N, wave->sample_rate);
                    sub_band_input_ch1[i] = differentiator(sub_band_input_ch1[i], N);
                    sub_band_input_ch1[i] = rectifier(sub_band_input_ch1[i], N);
                    current_bpm = comb_filter_convolution(sub_band_input_ch1[i], N, wave->sample_rate,
                                                          1, minbpm, maxbpm, high_limit);
                    if (current_bpm != -1) {
                        //printf("BPM computation is: %f\n", current_bpm);
                        frequency_map[(int) round(current_bpm)] += 1;
                    }

                    // Channel 2
                    //sub_band_input_ch2[i] = hanning_window(sub_band_input_ch2[i], N, wave->sample_rate);
                    sub_band_input_ch2[i] = differentiator(sub_band_input_ch2[i], N);
                    sub_band_input_ch2[i] = rectifier(sub_band_input_ch2[i], N);
                    current_bpm = comb_filter_convolution(sub_band_input_ch2[i], N, wave->sample_rate,
                                                          1, minbpm, maxbpm, high_limit);
                    if (current_bpm != -1) {
                        //printf("BPM computation is: %f\n", current_bpm);
                        frequency_map[(int) round(current_bpm)] += 1;
                    }

                    printf("Current BPM winner is: %i\n", most_frequent_bpm(frequency_map));

                }
            }

            winning_bpm = most_frequent_bpm(frequency_map);
            printf("BPM winner is: %i\n", winning_bpm);

            printf("\nDumping map...\n\n");
            dump_map(frequency_map);

            for (i = 0; i < num_sub_bands; i++) {
                free(sub_band_input_ch1[i]);
                free(sub_band_input_ch2[i]);
                free(sub_band_output_ch1[i]);
                free(sub_band_output_ch2[i]);
            }
            free(sub_band_input_ch1);
            free(sub_band_input_ch2);
            free(sub_band_output_ch1);
            free(sub_band_output_ch2);
            free(fft_input_ch1);
            free(fft_input_ch2);
            free(temp_data_buffer);
            free(data);
        } // if (size_is_correct) {
    } //  if (wave->format_type == 1) {

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
        hanning_in[i] = pow(cos(2*i*M_PI/hann_len), 2);
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

kiss_fft_scalar * rectifier(kiss_fft_scalar * input_buffer, unsigned int N) {
    /*
     * Rectifies a signal in the time domain.
     */
    int i;

    for (i = 1; i < N; i++) {
        if (input_buffer[i] < 0)
            input_buffer[i] *= -1;
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

double comb_filter_convolution(kiss_fft_scalar * data_input,
                               unsigned int N, unsigned int sample_rate, float resolution,
                               int minbpm, int maxbpm, int high_limit) {
    /*
     * Convolves the FFT of the data_input of size N with an impulse train
     * with a periodicity relative to the bpm in the range of minbpm to maxbpm.
     *
     * Returns bpm with the greatest measured energy
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

    int index;
    float i, bpm;
    unsigned int j, ti;
    unsigned int bpm_range = maxbpm - minbpm;
    double * energy;
    energy = (double *) malloc(sizeof(double)*(bpm_range/resolution));

    for (i = 0; i < bpm_range; (i+=resolution)) {

        // Ti is the period of impulses
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
        kiss_fftr(fft_cfg_data, data_input, data_output);

        filter_abs = absolute_value(filter_output, filter_abs, N);
        data_abs = absolute_value(data_output, data_abs, N);

        index = i/resolution;

        energy[index] = 0.0;

        for (j = 0; j < N/2; j++) {
            energy[index] += filter_abs[j] * data_abs[j];
        }

        //printf("Energy of bpm %f is %f\n", (minbpm+i), energy[index]);
    }

    bpm = (float)max_array(energy, bpm_range/resolution);

    if (bpm != -1) {
        bpm *= resolution;
        bpm += minbpm;
    }

    free(energy);
    free(filter_input);
    free(filter_output);
    free(filter_abs);
    free(data_output);
    free(data_abs);
    kiss_fft_cleanup();

    return bpm;
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
    //printf("Max value at %i is %f\n", index, max);
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
