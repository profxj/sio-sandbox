function [gwig,s,period,cpsi]=wvlt(g,dt,nvoice,k)
%function [gwig,s,period,cpsi]=wvlt(g,dt,nvoice,k) calculates 
%the wavelet transform gwig of the time series g.
%g is the input time series.  Should be a column vector.
%dt is the time step in user units.
%nvoice is the desired number of voices per octave.
%if nvoice is a vector, it is interpreted as the periods to be used
%in units of time steps.
%k is the parameter controlling the wavelet's Q.
%if k is not passed, it is assumed to be 1.  This is the usual call.
%gwig is the wavelet transform, a length(g) by length(s) matrix.
%s is the vector of scales.
%period is the vector of periods corresponding with s.
%cpsi is the admissibility coefficient.

%D. Rudnick, March 3, 1998.

N=length(g);
ghat=fft(g);
f=[(0:N/2) (-N/2+1:-1)]'/N;

if prod(size(nvoice)) == 1 
   s=2.^[1:1/nvoice:log2(N)]';
   s=[-flipud(s); s];
else
   s=nvoice(:);
end      

if nargin == 3, k=1; end;
psihatc=morlet(4*k*f,k);
s=k*s;
psihat=morlet(f*s',k);

cpsi=sum((abs(psihatc(2:N)).^2)./abs(f(2:N)))/N;
s=s*dt;
gwig=ifft((ghat*sqrt(abs(s))').*conj(psihat));
period=s/k;
