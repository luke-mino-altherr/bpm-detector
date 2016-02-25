#include <stdio.h>  
#include <stdlib.h>
#include <string.h>
#include "wave.h"
#include "kiss_fftr.h"
#include "BeatDetektor.h"

int i;


int main()
{
  char filename[] = "metronome.wav";

  WAVE * wave_file;
  wave_file = (WAVE *) malloc(sizeof(WAVE));
  read_wav(wave_file, filename);

  // N is the size of the data/fft array
  int n = 8 * wave_file->data_size / (wave_file->channels * wave_file->bits_per_sample);
  if (n%2 != 0)
    n++;

  // Sanity check
  /*printf("%i\n", wave_file->channels);
  printf("%i\n", wave_file->data_size);
  for (i=0;i<n/100;i++)
    printf("%f\n",wave_file->data[0][i]);*/

  kiss_fft_scalar zero;
  memset(&zero, 0, sizeof(zero));

  //kiss_fft_scalar * fft_input;
  //fft_input = (kiss_fft_scalar *) malloc(n * sizeof(kiss_fft_scalar));
  kiss_fft_cpx * fft_input_ch1;
  fft_input_ch1 = (kiss_fft_cpx *) malloc(n * sizeof(kiss_fft_cpx));
  kiss_fft_cpx * fft_input_ch2;
  fft_input_ch2 = (kiss_fft_cpx *) malloc(n * sizeof(kiss_fft_cpx));
  kiss_fft_cpx * fft_output_ch1;
  fft_output_ch1 = (kiss_fft_cpx *) malloc(n * sizeof(kiss_fft_cpx));
  kiss_fft_cpx * fft_output_ch2;
  fft_output_ch2 = (kiss_fft_cpx *) malloc(n * sizeof(kiss_fft_cpx));

  printf("Initializing fft input array\n");
  int i;
  for (i=0; i<n; i++) {
    fft_input_ch1[i].r = wave_file->data[0][i];
    fft_input_ch1[i].i = 0;
    fft_input_ch2[i].r = wave_file->data[1][i];
    fft_input_ch2[i].i = 0;
    fft_output_ch1[i].r = 0;
    fft_output_ch1[i].i = 0;
    fft_output_ch2[i].r = 0;
    fft_output_ch2[i].i = 0;
  }

  printf("Calculating FFT\n");
  //kiss_fftr_cfg fft_cfg_ch1 = kiss_fftr_alloc(n, 0, NULL, NULL);
  //kiss_fftr_cfg fft_cfg_ch2 = kiss_fftr_alloc(n, 0, NULL, NULL);
  //kiss_fftr(fft_cfg_ch1, fft_input, fft_output);

  kiss_fft_cfg fft_cfg_ch1 = kiss_fft_alloc(n, 0, NULL, NULL);
  kiss_fft_cfg fft_cfg_ch2 = kiss_fft_alloc(n, 0, NULL, NULL);
  kiss_fft(fft_cfg_ch1, fft_input_ch1, fft_output_ch1);
  kiss_fft(fft_cfg_ch2, fft_input_ch2, fft_output_ch2);

  // Write FFT data to file
  FILE * pFile;
  pFile = fopen("fft_ch1.dat", "wb");
  fwrite ((float *) fft_output_ch1, sizeof(float), sizeof(fft_output_ch1), pFile);
  fwrite ((float *) fft_output_ch2, sizeof(float), sizeof(fft_output_ch2), pFile);
  fclose (pFile);
  if (pFile)
    free(pFile);

/*
  printf("Plotting FFT");
  FILE *gnuplot = popen("gnuplot", "w");
  fprintf(gnuplot, "plot '-'\n");
  for (i = 0; i < n/10; i++)
    fprintf(gnuplot, "%g %g\n", (double) n, pow(pow(fft_output[i].r,2) + pow(fft_output[i].i,2), .5));
  fprintf(gnuplot, "e\n");
  fflush(gnuplot);
*/
  //transform((float *) fft_input, n/10);

  free(fft_input_ch1);
  free(fft_input_ch2);
  free(fft_output_ch1);
  free(fft_output_ch2);
  kiss_fft_free(fft_cfg_ch1);
  kiss_fft_free(fft_cfg_ch2);
  kiss_fft_cleanup();

  for (i=0; i<wave_file->channels; i++)
    free(wave_file->data[i]);

  free(wave_file);

  printf("done");

  return 0;

}
