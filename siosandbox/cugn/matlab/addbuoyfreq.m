function bindata=addbuoyfreq(bindata)
% bindata=addbuoyfreq(bindata) addes buoyancy frequency in cycles per hour
% to the bindata structure.

[~,dsigmadz]=gradient(bindata.sigma,bindata.depth(2)-bindata.depth(1));
dsigmadz(dsigmadz < 0)=0;
bindata.buoyfreq=sqrt(9.8/1025*dsigmadz)/(2*pi)*3600;