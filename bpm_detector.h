//
// Created by Luke Mino-Altherr on 2/24/16.
//

#ifndef BPM_DETECTOR_H
#define BPM_DETECTOR_H

typedef struct wave_file {
  unsigned char riff[4];                      // RIFF string
  unsigned int overall_size   ;               // overall size of file in bytes
  unsigned char wave[4];                      // WAVE string
  unsigned char fmt_chunk_marker[4];          // fmt string with trailing null char
  unsigned int length_of_fmt;                 // length of the format data
  unsigned int format_type;                   // format type. 1-PCM, 3- IEEE float, 6 - 8bit A law, 7 - 8bit mu law
  unsigned int channels;                      // no.of channels
  unsigned int sample_rate;                   // sampling rate (blocks per second)
  unsigned int byterate;                      // SampleRate * NumChannels * BitsPerSample/8
  unsigned int block_align;                   // NumChannels * BitsPerSample/8
  unsigned int bits_per_sample;               // bits per sample, 8- 8bits, 16- 16 bits etc
  unsigned char data_chunk_header[4];         // DATA string or FLLR string
  unsigned int data_size;                     // NumSamples * NumChannels * BitsPerSample/8 - size of the next chunk that will be read
} WAVE;

int most_frequent_bpm(int *);
void dump_map(int *);
int max_array(double *, int);
double * comb_filter_convolution(kiss_fft_scalar * data_input, double * energy,
                               unsigned int N, unsigned int sample_rate, float resolution,
                               int minbpm, int maxbpm, int high_limit);
kiss_fft_scalar * differentiator(kiss_fft_scalar * input_buffer, unsigned int N);
kiss_fft_scalar * half_wave_rectifier(kiss_fft_scalar * input_buffer, unsigned int N);
kiss_fft_scalar * full_wave_rectifier(kiss_fft_scalar * input_buffer, unsigned int N);
kiss_fft_scalar * hanning_window(kiss_fft_scalar * data_in, unsigned int N, unsigned int sampling_rate);
kiss_fft_scalar * absolute_value(kiss_fft_cpx * input, kiss_fft_scalar * output, unsigned int N);
kiss_fft_scalar ** filterbank(kiss_fft_scalar * time_data_in, kiss_fft_scalar ** filterbank, unsigned int N, unsigned int sampling_rate);
float compute_bpm(double * energy_buffer, unsigned int bpm_range, unsigned int minbpm, float resolution);
double * clear_energy_buffer(double * energy_buffer, unsigned int bpm_range, float resolution);
void plot(char *filename, kiss_fft_scalar *, kiss_fft_scalar *, int, unsigned int sampling_rate);
void plot2(char *filename, double *array1, int size);
void plot_complex(char*, kiss_fft_cpx *, int, unsigned int);
#endif // BPMDETECTOR_BPM_DETECTOR_H
