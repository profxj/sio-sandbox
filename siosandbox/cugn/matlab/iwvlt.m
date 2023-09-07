function g=iwvlt(gwig,s,nvoice,k)
%function g=iwvlt(gwig,s,nvoice,k) returns the inverse wavelet transform.
%if k is not passed, k=1 is assumed.
%

%D. Rudnick, March 4, 1998

N=size(gwig,1);
f=[(0:N/2) (-N/2+1:-1)]'/N;

if nargin == 3, k=1; end;

psihat=morlet(4*k*f,k);
cdelta=sum(conj(psihat(2:N))./abs(f(2:N)))/N;
g=sum(gwig.*(ones(N,1)*(1./sqrt(abs(s')))),2)/cdelta/nvoice*log(2);
