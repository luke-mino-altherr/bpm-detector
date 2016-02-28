//
// Created by Luke Mino-Altherr
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "kiss_fft.h"
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
            // Start reading audio data and computing transient beats / bpm
            /***********************************************************************************/

            unsigned int i = 0, j = 0, k = 0;
            char *tem_data_buffer;
            temp_data_buffer = (char *) calloc(bytes_in_each_channel, sizeof(char));

            // Used to recursively call comb filter bpm detection
            float a, b, c, d;

            int bpm_start = 60;
            int bpm_end = 180;
            int bpm_range = bpm_end - bpm_start;
            int winning_bpm = 0;

            int * frequency_map;
            frequency_map = (int *) calloc(bpm_range, sizeof(int));

            // Number of samples to analyze. We are taking 5 seconds of data.
            int N = 5*wave->sample_rate;

            double *abs_buffer_data, *abs_buffer_filter;
            abs_buffer_data = (double *) calloc(N, sizeof(double));
            abs_buffer_filter = (double *) calloc(N, sizeof(double));

            // Allocate history buffer to hold energy values for each tested BPM
            double *energy_history; double energy;
            energy_history = (double *) calloc(bpm_range, sizeof(double));

            // Data buffer to read from file into
            kiss_fft_scalar *data;
            data = (kiss_fft_scalar *) calloc(2, sizeof(kiss_fft_scalar));

            // Allocate FFT buffers
            kiss_fft_cfg fft_cfg_data = kiss_fft_alloc(N, 0, NULL, NULL);
            kiss_fft_cpx *fft_input_data, *fft_output_data;
            fft_input_data = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
            fft_output_data = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));

            // Initialize FFT buffers
            for (i = 0; i < N; i++) {
                fft_input_data[i].r = 0;
                fft_input_data[i].i = 0;
                fft_output_data[i].r = 0;
                fft_output_data[i].i = 0;
            }

            for (i = 0; i < floor(num_samples); i++) {
                    // Loop for left and right channels
                    for (k = 0; k < 2; k++) {
                        read = fread(data_buffer, sizeof(data_buffer), 1, ptr);

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

                    //printf("Data L: %f, Data R: %f\n", data[0], data[1]);

                    // Set data to FFT buffers
                    fft_input_data[j].r = (kiss_fft_scalar) data[0];
                    fft_input_data[j].i = (kiss_fft_scalar) data[1];
                }


                fft_input_data = rectifier(fft_input_data, N);
                fft_input_data = differentiator(fft_input_data, N);

                kiss_fft(fft_cfg_data, fft_input_data, fft_output_data);
                fft_output_data = hanning_window(fft_output_data, N, wave->sample_rate);

                // Compute FFT

                printf("Finsihed FFT\n");

                // Compute absolute value of FFT data.
                for (j = 0; j < N; j++) {
                    abs_buffer_data[j] = pow(pow(fft_output_data[j].r, 2) + pow(fft_output_data[j].i, 2), .5);
                }

                a = comb_filter_convolution(abs_buffer_data, N, bpm_start, bpm_end, high_limit, wave->sample_rate, 1);

                /*b = comb_filter_convolution(abs_buffer_data, N, a - 20, a + 20, high_limit,
                                            wave->sample_rate, .5);
                printf("Second bpm computation is: %f\n", b);
                c = comb_filter_convolution(abs_buffer_data, N, b - 5, b + 5, high_limit, wave->sample_rate, .1);
                printf("Third bpm computation is: %f\n", c);
                d = comb_filter_convolution(abs_buffer_data, N, c - 1, c + 1, high_limit, wave->sample_rate, .01);
                printf("Fourth bpm computation is: %f\n", d);*/

                if (a != -1) {
                    printf("BPM computation is: %f\n", a);
                    frequency_map[(int)round(a)] += 1;
                }
            }

            printf("BPM winner is: %i\n", most_frequent_bpm(frequency_map));


            //dump_map(frequency_map);
            free(data);
            free(abs_buffer_data);
            free(abs_buffer_filter);
            free(fft_input_data);
            free(fft_output_data);

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

kiss_fft_cpx * hanning_window(kiss_fft_cpx * input_buffer, unsigned int N, unsigned int sampling_rate) {
    /*
     * Implement a 200 ms half hanning window.
     *
     * Input is a signal in the frequency domain
     * Output is a windowed signal in the frequency domain.
     */
    kiss_fft_cfg fft_cfg = kiss_fft_alloc(N, 0, NULL, NULL);
    kiss_fft_cpx * hanning_in, *hanning_out;
    hanning_in = (kiss_fft_cpx *) malloc(N*sizeof(kiss_fft_cpx));
    hanning_out = (kiss_fft_cpx *) malloc(N*sizeof(kiss_fft_cpx));

    int hann_len = .2*sampling_rate;

    int i;
    for (i = 0; i < N; i++) {
        hanning_in[i].r = pow(cos(i*M_PI/hann_len/2),2);
        hanning_in[i].i = 0;
        hanning_out[i].r = 0;
        hanning_out[i].i = 0;
    }

    kiss_fft(fft_cfg, hanning_in, hanning_out);

    for (i = 0; i < N; i++) {
        input_buffer[i].r *= hanning_out[i].r;
        input_buffer[i].i *= hanning_out[i].i;
    }

    free (hanning_in); free(hanning_out);

    return input_buffer;
}

kiss_fft_cpx * rectifier(kiss_fft_cpx * input_buffer, unsigned int N) {
    /*
     * Rectifies a signal in the time domain.
     */
    int i;

    for (i = 1; i < N; i++) {
        if (input_buffer[i].r < 0)
            input_buffer[i].r *= -1;
        if (input_buffer[i].i < 0)
            input_buffer[i].i *= -1;
    }

    return input_buffer;
}

kiss_fft_cpx * differentiator(kiss_fft_cpx * input_buffer, unsigned int N) {
    /*
     * Differentiates a signal in the time domain.
     */
    kiss_fft_cpx * output;
    output = (kiss_fft_cpx *) malloc(N*sizeof(kiss_fft_cpx));

    int i;
    output[0].r = 0;
    output[0].i = 0;

    for (i = 1; i < N; i++) {
        output[i].r = input_buffer[i].r - input_buffer[i-1].r;
        output[i].i = input_buffer[i].i - input_buffer[i-1].i;
    }

    for (i = 0; i < N; i++) {
        input_buffer[i].r = output[i].r;
        input_buffer[i].i = output[i].i;
    }

    free(output);

    return input_buffer;
}

double comb_filter_convolution(double * abs_buffer, unsigned int N, int minbpm, int maxbpm, int high_limit,
                               unsigned int sample_rate, float resolution) {
    /*
     * Convolves the abs_buffer of size N with a impulse train
     * for the bpm range, minbpm to maxbpm.
     *
     * Returns bpm with the greatest measured energy/
     */
    kiss_fft_cfg fft_cfg_filter = kiss_fft_alloc(N, 0, NULL, NULL);
    kiss_fft_cpx *fft_input_filter, *fft_output_filter;
    fft_input_filter = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));
    fft_output_filter = (kiss_fft_cpx *) malloc(N * sizeof(kiss_fft_cpx));

    int index;
    float i, bpm;
    unsigned int j, ti;
    unsigned int bpm_range = maxbpm - minbpm;
    double * energy, *abs_buffer_filter;
    energy = (double *) malloc(sizeof(double)*(bpm_range/resolution));
    abs_buffer_filter = (double *) malloc(sizeof(double) * N);

    printf("! ");

    for (i = 0; i < bpm_range; (i+=resolution)) {

        // Ti is the period of impulses
        ti = (double)60/(minbpm + i)*sample_rate;

        //printf("ti: %i; bpm: %f\n", ti, (minbpm+i));

        for (j = 0; j < N; j++) {
            if (j%ti == 0) {
                fft_input_filter[j].r = (kiss_fft_scalar) high_limit;
                fft_input_filter[j].i = (kiss_fft_scalar) high_limit;
                //printf("%f, %u, %u \n", i, j, ti);
            } else {
                fft_input_filter[j].r = 0.0;
                fft_input_filter[j].i = 0.0;
            }
        }

        kiss_fft(fft_cfg_filter, fft_input_filter, fft_output_filter);

        index = i/resolution;

        energy[index] = 0.0;

        for (j = 0; j < N; j++) {
            abs_buffer_filter[j] = pow(pow(fft_output_filter[j].r, 2) + pow(fft_output_filter[j].i, 2), .5);
            energy[index] += abs_buffer[j] * abs_buffer_filter[j];
        }

        //printf("Energy of bpm %f is %f\n", (minbpm+i), energy[index]);
    }

    bpm = (float)maxArray(energy, bpm_range/resolution);

    if (bpm != -1) {
        bpm *= resolution;
        bpm += minbpm;
    } else {
        bpm = -1;
    }

    free(energy); free(abs_buffer_filter);
    free(fft_input_filter); free(fft_output_filter);

    return bpm;
}

int maxArray(double * array, int size) {
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
    int i, winner=0;
    for (i=0; i<200; i++) {
        if (map[i] > winner)
            winner = i;
    }
    return winner;
}
