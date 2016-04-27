function output=control(song1, loops)

% CONTROL takes in the names of a .wav file, and outputs the bpm.
%
%     SIGNAL = CONTROL(SONG1, LOOPS) takes in
%     the names of a .wav file as a string and the number of 5 second
%     chunks to process when computing the BPM. 

%     Defaults are:
%        BANDLIMITS = [0 200 400 800 1600 3200]
  % Takes in the wave file
  tic;
  song_path = strcat(pwd,'/',song1);
  [x1, fs, nbits, opts] = wavread(song_path);
  opts.fmt

  left = x1(:,1);
  right = x1(:,2);
 
  start = 1;
  stop = fs * 5;

  maxfreq = fs/2;
  minbpm = 60;
  maxbpm = 180;
  bpm_map = zeros(1, maxbpm);
  
  for i = 1:loops
    sample_left = left(start:stop);
    sample_right = right(start:stop);
  
    % Implements beat detection algorithm for each song
  
    status = 'filtering first song...';
    a1 = filterbank(sample_left, fs);
    a2 = filterbank(sample_right, fs);
    status = 'windowing first song...';
    b1 = hwindow(a1, fs);
    b2 = hwindow(a2, fs);
    status = 'differentiating first song...';
    c1 = diffrect(b1, 6);
    c2 = diffrect(b2, 6);
    status = 'comb filtering first song...';
    d1 = timecomb(c1, 1, minbpm, maxbpm, fs);
    d2 = timecomb(c2, 1, minbpm, maxbpm, fs);
  
    bpm_map(d1) = bpm_map(d1) + 1;
    bpm_map(d2) = bpm_map(d2) + 1;
    
    start = stop;
    stop = stop + fs*5;
    if start >= length(left)
        break;
    end
    if stop > length(left)
      stop = length(left);
    end
  end
  
  [~, max_bpm] = max(bpm_map);
  fprintf('Winning BPM is %i\n', max_bpm)
  
  toc
