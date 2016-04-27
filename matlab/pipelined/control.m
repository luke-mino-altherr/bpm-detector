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
  
  spmd
      for i = 1:loops
        if labindex == 1
            sample_left = left(start:stop);
            sample_right = right(start:stop);
            labSend(sample_left, 2);
            labSend(sample_right, 3);
            start = stop;
            stop = stop + fs*5;
            if start >= length(left)
                i = loops;
            end
            if stop > length(left)
                stop = length(left);
            end      
        end
      
      
        if labindex == 2 || labindex == 3
            a = labReceive(1);
            status = 'filtering first song...';
            b = filterbank(a, fs);
            labSend(b, labindex+2);
        end
        
        if labindex == 4 || labindex == 5
            c = labReceive(labindex-2);
            status = 'windowing first song...';
            d = hwindow(c, fs);
            labSend(d, labindex + 2);
        end
        
        if labindex == 6 || labindex == 7
            e = labReceive(labindex-2);
            status = 'differentiating first song...';
            f = diffrect(e);
            labSend(f, labindex+2);
        end
        
        if labindex == 8 || labindex == 9
            g = labReceive(labindex-2);
            status = 'comb filtering first song...';
            h = timecomb(g, 1, minbpm, maxbpm, fs);
            labSend(h, 10);
        end
        
        if labindex == 10
            bpm1 = labReceive(8);
            bpm2 = labReceive(9);
            bpm_map(bpm1) = bpm_map(bpm1) + 1;
            bpm_map(bpm2) = bpm_map(bpm2) + 1;
            
            if i == loops
                [~, max_bpm] = max(bpm_map);
                fprintf('Winning BPM is %i\n', max_bpm)
            end
                
        end
        
      end
      
  end
  
  
  
  toc
