function [dist,offset] = ll2do(lon,lat,lon0,lat0,lon1,lat1)

% Calculates distance given start and end of line lon0, lat0, lon1, lat1 (lons and
% lats)
%
% D. Rudnick, 28 May 2018 

% Set parameters
nm2km = 1.852;
deg2min = 60;
deg2rad = pi/180;
deg2km=deg2min*nm2km;

% Calculate angle of new coordinate system relative to east
dyy = (lat1-lat0);
dxx = cos(1/2*(lat1+lat0)*deg2rad)*(lon1-lon0);
theta = atan2(dyy,dxx);

% Calculate x, y of lon, lat relative to start of line
dy = (lat-lat0)*deg2km;
dx = cos(1/2*(lat1+lat0)*deg2rad).*(lon-lon0)*deg2km;

% Calculate dist, offset in new coordinate system by rotating
z=dx+1i*dy;
zhat=z*exp(-1i*theta);

dist=real(zhat);
offset=imag(zhat);
