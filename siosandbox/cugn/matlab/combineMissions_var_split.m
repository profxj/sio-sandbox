function ctd = combineMissions_var_split(line,vars,yearstart,yearend)

% Concatenate data from all missions along a particular CalCOFI line. Use variables 
% in cell array vars. Input parameters are:
% line: '66.7','80.0', or '90.0', or 'along'
    % vars: cell array of variables
%
% K.Zaba, May22,2014
% D. Rudnick, September 8, 2016 - update to use cugn.txt
% D. Rudnick, March 12, 2018 - update to use vars and data in binmat
% N. Anidjar, January 26, 2021 - update to include option for alongshore
% line.

% Paths
list = '~rudnick/glider/data/deployments/cugn2.txt';

% CTD Variables
ctd1D = {'time','lon','lat','dist','offset', ...
       'timeu','lonu','latu','distu','offsetu','u','v'};

ctd2D = vars;
nctd1D = length(ctd1D); 
nctd2D = length(ctd2D); 
depthLevels = (10:10:500)'; 
nz = length(depthLevels);

% Line definitions
ctd.line=line;
if strcmp(line(1:2),'66')
   ctd.lonendpts = [-121.8371; -124.2000];
   ctd.latendpts = [36.8907; 35.7900];
elseif strcmp(line(1:2),'80')
   ctd.lonendpts = [-120.4773;-123.9100];
   ctd.latendpts = [34.4703; 32.8200];
elseif strcmp(line(1:2),'90')
   ctd.lonendpts = [-117.7475; -124.0000];
   ctd.latendpts = [33.5009; 30.4200];
elseif strcmp(line(1:2),'al')
   ctd.lonendpts = [-119.9593; -121.1500];
   ctd.latendpts = [32.4179; 34.1500];
end

ctd.yearend=yearend;
ctd.yearstart=yearstart;

% Initialize CTD Output 
ctd.depth = depthLevels;
for iictd1D = 1:nctd1D
    ctd.(ctd1D{iictd1D}) = [];
end
for iictd2D = 1:nctd2D
    ctd.(ctd2D{iictd2D}) = [];
end

% ReadIn Mission Data
fid=fopen(list);
deps = textscan(fid,'%s %s %f %f','Delimiter',',');
fclose(fid);
ii=find(strncmp(line,deps{2},2));
ctd.missions=deps{1}(ii);
dives = [deps{3}(ii) deps{4}(ii)]; % make an array with start and end dives of a line.

for idep=1:length(ii) % looping through deployments along the selected line.
   filename=[ctd.missions{idep} '_bin.mat'];
   if exist(filename,'file')
      load(filename,'bindata');
   else
      load(ctd.missions{idep},'bindata');
   end
   
   [bindata.dist,bindata.offset]   = ll2do(bindata.lon, bindata.lat, ...
       ctd.lonendpts(1),ctd.latendpts(1),ctd.lonendpts(2),ctd.latendpts(2));
   [bindata.distu,bindata.offsetu] = ll2do(bindata.lonu, bindata.latu, ...
       ctd.lonendpts(1),ctd.latendpts(1),ctd.lonendpts(2),ctd.latendpts(2));
   
   dive = dives(idep,:);
   for iictd1D=1:nctd1D % for each of the 1D variables...
       if ~isnan(dive(1)) % if dive numbers are specified,
           if isinf(dive(2))
               ctd.(ctd1D{iictd1D}) = [ctd.(ctd1D{iictd1D}); bindata.(ctd1D{iictd1D})(dive(1):end)];
           else
                ctd.(ctd1D{iictd1D}) = [ctd.(ctd1D{iictd1D}); bindata.(ctd1D{iictd1D})(dive(1):dive(2))];
           end
       else
           ctd.(ctd1D{iictd1D}) = [ctd.(ctd1D{iictd1D}); bindata.(ctd1D{iictd1D})];
       end
   end
   
   for iictd2D=1:nctd2D % for each of the 2D variables...
       if ~isnan(dive(1)) % if dive numbers are specified,
           if isinf(dive(2))
               if isfield(bindata,ctd2D{iictd2D})% if it is an existing field in bindata...
                   ctd.(ctd2D{iictd2D})=[ctd.(ctd2D{iictd2D}) bindata.(ctd2D{iictd2D})(1:nz,dive(1):end)];
                   % add to ctd struct.
               else % else (if field does not exist)...
                   ctd.(ctd2D{iictd2D})=[ctd.(ctd2D{iictd2D}) nan(size(bindata.t(1:nz,dive(1):end)))];
                   % add nan array to ctd struct.
               end
           else
               if isfield(bindata,ctd2D{iictd2D})% if it is an existing field in bindata...
                   ctd.(ctd2D{iictd2D})=[ctd.(ctd2D{iictd2D}) bindata.(ctd2D{iictd2D})(1:nz,dive(1):dive(2))];
                   % add to ctd struct.
               else % else (if field does not exist)...
                   ctd.(ctd2D{iictd2D})=[ctd.(ctd2D{iictd2D}) nan(size(bindata.t(1:nz,dive(1):dive(2))))];
                   % add nan array to ctd struct.
               end
           end
       else % if dive numbers are not specified
           if isfield(bindata,ctd2D{iictd2D})% if the 2D variable is an existing field in bindata...
               ctd.(ctd2D{iictd2D})=[ctd.(ctd2D{iictd2D}) bindata.(ctd2D{iictd2D})(1:nz,:)];
               % add to ctd struct.
           else % else (if the 2D variable does not exist)...
               ctd.(ctd2D{iictd2D})=[ctd.(ctd2D{iictd2D}) nan(size(bindata.t(1:nz,:)))];
               % add nan array to ctd struct.
           end
       end
   end
      
      clear('bindata');
end

%prune to yearstart, yearend
dv=datevec(ut2dn(ctd.time));
iitime=dv(:,1) >= yearstart & dv(:,1) <= yearend;
for iictd1D=1:nctd1D % for each of the 1D variables...
   ctd.(ctd1D{iictd1D})=ctd.(ctd1D{iictd1D})(iitime);
end
for iictd2D=1:nctd2D % for each of the 2D variables...
   ctd.(ctd2D{iictd2D})=ctd.(ctd2D{iictd2D})(:,iitime);
end
