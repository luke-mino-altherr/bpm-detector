function output = filterbank(sig, bandlimits, maxfreq)

% FILTERBANK divides a time domain signal into individual frequency
% bands.
%     
%     FREQBANDS = FILTERBANK(SIG, BANDLIMITS, MAXFREQ) takes in a
%     time domain signal stored in a column vector, and outputs a
%     vector of the signal in the frequency domain, with each
%     column representing a different band. BANDLIMITS is a vector
%     of one row in which each element represents the frequency
%     bounds of a band. The final band is bounded by the last
%     element of BANDLIMITS and  MAXFREQ. 
%
%     Defaults are:
%        BANDLIMITS = [0 200 400 800 1600 3200]
%        MAXFREQ = 4096
%
%     This is the first step of the beat detection sequence.
%
%     See also HWINDOW, DIFFRECT, and TIMECOMB

  if nargin < 2, bandlimits=[0 200 400 800 1600 3200]; end
  if nargin < 3, maxfreq=4096; end

  dft = fft(sig);

  n = length(dft);
  nbands = length(bandlimits);
  
  % Bring band scale from Hz to the points in our vectors 
  
  for i = 1:nbands-1
    bl(i) = floor(bandlimits(i)/maxfreq*n/2)+1;
    br(i) = floor(bandlimits(i+1)/maxfreq*n/2);
  end

  bl(nbands) = floor(bandlimits(nbands)/maxfreq*n/2)+1;
  br(nbands) = floor(n/2);

  output = zeros(n,nbands);

  % Create the frequency bands and put them in the vector output.
  
  for i = 1:nbands
    output(bl(i):br(i),i) = dft(bl(i):br(i));
    output(n+1-br(i):n+1-bl(i),i) = dft(n+1-br(i):n+1-bl(i));
  end
 
  output(1,1)=0;










