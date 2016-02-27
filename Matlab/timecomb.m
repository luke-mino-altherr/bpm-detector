function output = timecomb(sig, acc, minbpm, maxbpm, bandlimits, maxfreq)

% TIMECOMB finds the tempo of a musical signal, divided into
% frequency bands.
%
%     BPM = TIMECOMB(SIG, ACC, MINBPM, MAXBPM, BANDLIMITS, MAXFREQ) 
%     takes in a vector containing a signal, with each band stored
%     in a different column. BANDLIMITS is a vector of one row in
%     which each element represents the frequency bounds of a
%     band. The final band is bounded by the last element of
%     BANDLIMITS and MAXFREQ. The beat resolution is defined in
%     ACC, and the range of beats to test is  defined by MINBPM and
%     MAXBPM. 
%  
%     Defaults are:
%        ACC = 1
%        MINBPM = 60
%        MAXBPM = 240
%        BANDLIMITS = [0 200 400 800 1600 3200]
%        MAXFREQ = 4096
%
%     Note that timecomb can be recursively called with greater
%     accuracy and a smaller range to speed up computation.    
%
%     This is the last step of the beat detection sequence.
%
%     See also FILTERBANK, HWINDOW, and DIFFRECT
  
  if nargin < 2, acc = 1; end 
  if nargin < 3, minbpm = 60; end
  if nargin < 4, maxbpm = 240; end
  if nargin < 5, bandlimits = [0 200 400 800 1600 3200]; end
  if nargin < 6, maxfreq = 4096; end


  n=length(sig);

  nbands=length(bandlimits);

  % Set the number of pulses in the comb filter
  
  npulses = 3;

  % Get signal in frequency domain

  for i = 1:nbands
    dft(:,i)=fft(sig(:,i));
  end
  
  % Initialize max energy to zero
  
  maxe = 0;
  
  for bpm = minbpm:acc:maxbpm
    
    % Initialize energy and filter to zero(s)
    
    e = 0;
    fil=zeros(n,1);
    
    % Calculate the difference between peaks in the filter for a
    % certain tempo
    
    nstep = floor(120/bpm*maxfreq);
    
    % Print the progress
    
    percent_done  = 100*(bpm-minbpm)/(maxbpm-minbpm)
    
    % Set every nstep samples of the filter to one
    
    for a = 0:npulses-1
      fil(a*nstep+1) = 1;
    end
    
    % Get the filter in the frequency domain
    
    dftfil = fft(fil);
    
    % Calculate the energy after convolution
    
    for i = 1:nbands
      x = (abs(dftfil.*dft(:,i))).^2;
      e = e + sum(x);
    end
    
    % If greater than all previous energies, set current bpm to the
    % bpm of the signal
    
    if e > maxe
      sbpm = bpm;
      maxe = e;
    end
  end
  
  output = sbpm;
  
  






