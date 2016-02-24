// From http://truelogic.org/wordpress/2015/09/04/parsing-a-wav-file-in-c/

/**
 * Read and parse a wave file
 *
 **/
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wave.h"
#define TRUE 1
#define FALSE 0

unsigned char buffer4[4];
unsigned char buffer2[2];

//char* seconds_to_time(float seconds);

FILE *ptr;

void read_wav(WAVE * wave, char * filename) {

    // get file path
    char * cwd = NULL;
    cwd = (char *) malloc(1024*sizeof(char));
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        strcat(cwd, "/");
        strcat(cwd, filename);
        printf("%s\n", cwd);
    }

    // open file
    printf("Opening  file..\n");
    ptr = fopen(filename, "r");
    if (ptr == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    int read = 0;

    // read wave parts

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


    // calculate no.of samples
    long num_samples = (8 * wave->data_size) / (wave->channels * wave->bits_per_sample);
    printf("Number of samples:%lu \n", num_samples);

    long size_of_each_sample = (wave->channels * wave->bits_per_sample) / 8;
    printf("Size of each sample:%ld bytes\n", size_of_each_sample);

    // calculate duration of file
    float duration_in_seconds = (float) wave->overall_size / wave->byterate;
    printf("Approx.Duration in seconds=%f\n", duration_in_seconds);
    //printf("Approx.Duration in h:m:s=%s\n", seconds_to_time(duration_in_seconds));

    // read each sample from data chunk if PCM
    if (wave->format_type == 1) { // PCM

        long i =0;
        int cnt = 0;
        char data_buffer[size_of_each_sample];
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
            printf("\n\n.Valid range for data values : %ld to %ld \n", low_limit, high_limit);

            wave->data = (float **) malloc(wave->channels*sizeof(float *));
            if(wave->data == NULL) {
                fprintf(stderr, "out of memory\n");
                exit(1);
            }
            for (i = 0; i < wave->channels; i++)
                wave->data[i] = (float *) malloc(num_samples*sizeof(float));

            for (i = 0; i < num_samples; i++) {
                //printf("==========Sample %ld / %ld=============\n", i, num_samples);
                read = fread(data_buffer, sizeof(data_buffer), 1, ptr);
                if (read == 1) {

                    // dump the data read
                    unsigned int xchannels = 0;
                    float data = 0;

                    for (xchannels = 0; xchannels < wave->channels; xchannels ++ ) {
                        //printf("Channel#%d : ", (xchannels+1));
                        // convert data from little endian to big endian based on bytes in each channel sample
                        if (bytes_in_each_channel == 4) {
                            data = data_buffer[0] |
                                                       (data_buffer[1]<<8) |
                                                       (data_buffer[2]<<16) |
                                                       (data_buffer[3]<<24);
                            //printf("%f", data);
                            wave->data[xchannels][i] = data;
                            //memset(wave->data[xchannels], data, sizeof(wave->data[xchannels][0]*i));
                            //memcpy(wave->data[xchannels], &data, sizeof(data));
                        }
                        else if (bytes_in_each_channel == 2) {
                            data = data_buffer[0] |
                                                       (data_buffer[1] << 8);
                            wave->data[xchannels][i] = data;
                            //memcpy(wave->data[xchannels], &data, sizeof(data));

                        }
                        else if (bytes_in_each_channel == 1) {
                            data = data_buffer[0];
                            wave->data[xchannels][i] = data;
                            //memset(wave->data[xchannels], data, sizeof(wave->data[xchannels][0]*i));
                            //memcpy(wave->data[xchannels], &data, sizeof(data));
                        }

                        // check if value was in range
                        if (wave->data[xchannels][i] < low_limit || wave->data[xchannels][i] > high_limit)
                            printf("**value out of range\n");

                        //printf(" | ");
                    }

                    //printf("\n");
                }
                else {
                    printf("Error reading file. %d bytes\n", read);
                    break;
                }

            } // for (i =1; i <= num_samples; i++) {

        } // if (size_is_correct) {
    } //  if (wave->format_type == 1) {

    if (ptr) {
        printf("Closing file..\n");
        fclose(ptr);
        ptr = NULL;
    }

    free(cwd);
}

/**
 * Convert seconds into hh:mm:ss format
 * Params:
 *seconds - seconds value
 * Returns: hms - formatted string
 **/
char* seconds_to_time(float raw_seconds) {
    char *hms;
    int hours, hours_residue, minutes, seconds, milliseconds;
    hms = (char*) malloc(100);

    sprintf(hms, "%f", raw_seconds);

    hours = (int) raw_seconds/3600;
    hours_residue = (int) raw_seconds % 3600;
    minutes = hours_residue/60;
    seconds = hours_residue % 60;
    milliseconds = 0;

    // get the decimal part of raw_seconds to get milliseconds
    char *pos;
    pos = strchr(hms, '.');
    int ipos = (int) (pos - hms);
    char decimalpart[15];
    memset(decimalpart, ' ', sizeof(decimalpart));
    strncpy(decimalpart, &hms[ipos+1], 3);
    milliseconds = atoi(decimalpart);


    sprintf(hms, "%d:%d:%d.%d", hours, minutes, seconds, milliseconds);
    return hms;
}
