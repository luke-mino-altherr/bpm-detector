function output=control(song1, song2, bandlimits, maxfreq)

% CONTROL takes in the names of two .wav files, and outputs their
% combination, beat-matched, and phase aligned.
%
%     SIGNAL = CONTROL(SONG1, SONG2, BANDLIMITS, MAXFREQ) takes in
%     the names of two .wav files, as strings, and outputs their
%     sum. BANDLIMITS and MAXFREQ are used to divide the signal for
%     beat-matching
%
%     Defaults are:
%        BANDLIMITS = [0 200 400 800 1600 3200]
%        MAXFREQ = 4096
  
  if nargin < 3, bandlimits = [0 200 400 800 1600 3200]; end
  if nargin < 4, maxfreq = 4096; end
  
  % Length (in samples) of 5 seconds of the song
  
  sample_size = floor(2.2*2*maxfreq); 
  
  % Takes in the two wave files
  
  x1 = wavread(strcat('net/screech/rpverret/elec301/',song1, '.wav'));
  x2 = wavread(strcat('net/screech/rpverret/elec301/',song2, '.wav'));
  
  % Differentiates between the shorter and longer signal
  
  if length(x1) < length(x2)
    short_song = x1;
    long_song = x2;
    short_length = length(x1);
  else
    short_song = x2;
    long_song = x1;
    short_length = length(x2);
  end
 
  start = floor(short_length/2 - sample_size/2)
  stop = floor(short_length/2 + sample_size/2)
  
  % Finds a 5 second representative sample of each song
  
  short_sample = short_song(start:stop);
  long_sample = long_song(start:stop);
  
  % Implements beat detection algorithm for each song
  
  status = 'filtering first song...'
  a = filterbank(short_sample, bandlimits, maxfreq);
  status = 'windowing first song...'
  b = hwindow(a, 0.2, bandlimits, maxfreq);
  status = 'differentiating first song...'
  c = diffrect(b, length(bandlimits));
  status = 'comb filtering first song...'
  
  % Recursively calls timecomb to decrease computational time
  
  d = timecomb(c, 2, 60, 240, bandlimits, maxfreq);
  e = timecomb(c, .5, d-2, d+2, bandlimits, maxfreq);
  f = timecomb(c, .1, e-.5, e+.5, bandlimits, maxfreq);
  g = timecomb(c, .01, f-.1, f+.1, bandlimits, maxfreq);
  
  short_song_bpm = g;
  
  status = 'filtering second song...'
  a = filterbank(long_sample, bandlimits, maxfreq);
  status = 'windowing second song...'
  b = hwindow(a, 0.2, bandlimits, maxfreq);
  status = 'differentiating second song...'
  c = diffrect(b, length(bandlimits));
  status = 'comb filtering second song...'
  d = timecomb(c, 2, 60, 240, bandlimits, maxfreq);
  e = timecomb(c, .5, d-2, d+2, bandlimits, maxfreq);
  f = timecomb(c, .1, e-.5, e+.5, bandlimits, maxfreq);
  g = timecomb(c, .01, f-.1, f+.1, bandlimits, maxfreq);
  
  long_song_bpm = g;
  
  % Finds the closest multiple of the slower tempo song to the
  % faster tempo song
  
  if (short_song_bpm > long_song_bpm)
    multiple = short_song_bpm / long_song_bpm;
    long_song_bpm
    if (abs(short_song_bpm - floor(multiple)*long_song_bpm)> ...
	abs(short_song_bpm - ceil(multiple)*long_song_bpm))
      new_long_song_bpm = ceil(multiple)*long_song_bpm
    else
      new_long_song_bpm = floor(multiple)*long_song_bpm
    end
    short_song_bpm
    new_short_song_bpm = short_song_bpm
  else
    multiple = long_song_bpm / short_song_bpm;
    short_song_bpm
    if (abs(long_song_bpm - floor(multiple)*short_song_bpm) > ...
	abs(long_song_bpm - ceil(multiple)*short_song_bpm))
      new_short_song_bpm = ceil(multiple)*short_song_bpm
    else
      new_short_song_bpm = floor(multiple)*short_song_bpm
    end
    long_song_bpm
    new_long_song_bpm = long_song_bpm
  end
  
  % Scales one signal so that the beats match
  
  if new_long_song_bpm > new_short_song_bpm
    final_bpm = new_long_song_bpm
    short_song = timescale(short_song, 1 - (new_long_song_bpm - new_short_song_bpm)/new_long_song_bpm);
  else
    final_bpm = new_short_song_bpm
    long_song = timescale(long_song, 1 - (new_short_song_bpm - new_long_song_bpm)/new_short_song_bpm);
  end
  
  % Finds the location of the first beat of each signal
  
  short_first_beat = phasealign(short_song, final_bpm, bandlimits,  maxfreq)
  long_first_beat = phasealign(long_song, final_bpm, bandlimits, maxfreq)
  
  diff = long_first_beat - short_first_beat
  
  % Zero-pads the beginning of one signal so the beats are in phase
  
  if diff > 0
    short_song = [zeros(diff,1); short_song];
  else
    long_song = [zeros(-diff,1); long_song];
  end
  
  % Zero-pads the end of one signal so that they are the same length
  
  if (length(short_song) > length(long_song))
    diff = length(short_song) - length(long_song);
    long_song = [long_song; zeros(diff,1)];
  else
    diff = length(long_song) - length(short_song);
    short_song = [short_song; zeros(diff,1)];
  end

  output = short_song + long_song;
 
  % Computes the scale necessary to scale each song to a maximum
  % amplitude of one
  
  h = max(output);
  l = - min(output);
  scale = max(h,l);
  
  % Scales output
  
  output = output/scale;

