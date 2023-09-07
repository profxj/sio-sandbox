function vari=anncycinterp(A,variable,level,time,dist)

vari=nan(size(time));

timebin=2*pi*time/86400/365.25;
variStr1D = {'u','v'};          % 1D variables
variStr2D = {'t','s','fl','abs','udop','vdop','ox','oxumolkg'};     % 2D variables
if ismember(variable,variStr2D)
   maxharmonic=size(A.(variable).sin,3);
   G=[ones(size(timebin)) sin(timebin*(1:maxharmonic)) cos(timebin*(1:maxharmonic))];
   
   ii=dist <= min(A.xcenter);
   mm=[A.(variable).constant(level,1); squeeze(A.(variable).sin(level,1,:)); squeeze(A.(variable).cos(level,1,:))];
   vari(ii)=G(ii,:)*mm;
   
   jj=dist >= max(A.xcenter);
   mm=[A.(variable).constant(level,end); squeeze(A.(variable).sin(level,end,:)); squeeze(A.(variable).cos(level,end,:))];
   vari(jj)=G(jj,:)*mm;
   
   dx=diff(A.xcenter);
   kk=find(~ii & ~jj & ~isnan(dist));
   for n=1:length(kk)
      xx=A.xcenter-dist(kk(n));
      np=find(xx(1:end-1).*xx(2:end) <= 0);
      
      mm=[A.(variable).constant(level,np:np+1); squeeze(A.(variable).sin(level,np:np+1,:))'; squeeze(A.(variable).cos(level,np:np+1,:))'];
      bracket=G(kk(n),:)*mm;
      vari(kk(n))=bracket*[xx(np+1); -xx(np)]/dx(np);
   end
   
elseif ismember(variable,variStr1D)
   maxharmonic=size(A.(variable).sin,2);
   G=[ones(size(timebin)) sin(timebin*(1:maxharmonic)) cos(timebin*(1:maxharmonic))];
   
   ii=dist <= min(A.xcenter);
   mm=[A.(variable).constant(1); A.(variable).sin(1,:)'; A.(variable).cos(1,:)'];
   vari(ii)=G(ii,:)*mm;
   
   jj=dist >= max(A.xcenter);
   mm=[A.(variable).constant(end); A.(variable).sin(end,:)'; A.(variable).cos(end,:)'];
   vari(jj)=G(jj,:)*mm;
   
   dx=diff(A.xcenter);
   kk=find(~ii & ~jj & ~isnan(dist));
   for n=1:length(kk)
      xx=A.xcenter-dist(kk(n));
      np=find(xx(1:end-1).*xx(2:end) <= 0);
      
      mm=[A.(variable).constant(np:np+1)'; A.(variable).sin(np:np+1,:)'; A.(variable).cos(np:np+1,:)'];
      bracket=G(kk(n),:)*mm;
      vari(kk(n))=bracket*[xx(np+1); -xx(np)]/dx(np);
   end
   
else
   error('No matching variable in A')
end
