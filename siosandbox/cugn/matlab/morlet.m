function psihat=morlet(f,k)
%function psihat=morlet(f,k) returns the Fourier transform of the
%Morlet wavelet given the frequency f.
%k is the parameter that sets the number of wiggles inside the
%Gaussian.
%

%D. Rudnick, March 3, 1998

psihat=zeros(size(f));
ii=find(f > 0);
psihat(ii)=sqrt(2*pi)*exp(-((2*pi*(k-f(ii))).^2)/2);

