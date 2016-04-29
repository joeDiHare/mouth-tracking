function output = lipTracker(filepath,flag_plot)

%%% script to perform computation of mouth area
%%% [SC] Apr-16
if nargin<2, flag_plot=false; end    

pathV=filepath;
pathA=[strrep(pathV,'video','audio'),'.wav'];

items = dir(pathV); items = items(3:end);
l_f=length(items); %no. of frames
% (0) get audio RMS
[sig,fs]=wavread(pathA); sig=sig./max(sig);
N=length(sig)/l_f;
for n=1:l_f, Rms(n) = rms(sig(1+(n-1)*N:n*N)); end
t=linspace(0,length(sig)/fs,l_f);

% % (1) manually select mouth area
% Selecting the two vartices of a rectangle will select the mouth area
% (Note that the actual area that is used is an ellipse)
% If the first selection is not accurate, repeat the process without 
% closing the figure. A new square will appear.
close(figure(1));
im1 = imread(strcat(pathV,'\',items(1).name));   
im1d= double(im1(:,:,:));
% f1=figure(1); 
% imshow(im1); title 'SELECT MOUTH AREA';
% set(f1, 'Position', get(0,'Screensize')); 
% global ru; ru=[];
% dcm_obj = datacursormode(gcf);
% datacursormode on;
% set(dcm_obj,'UpdateFcn', @myupdatefcn)
% waitfor(f1);
% x = min(ru(end-1,1),ru(end,1)); y = min(ru(end-1,2),ru(end,2));
% w = abs(ru(end-1,1)-ru(end,1)); h = abs(ru(end-1,2)-ru(end,2)); 
x =230;y=203;w=58;h=19;%comment this line
centroid=[x+w/2; y+h/2];     

xON = x; xOFF=x+w; HorizontalAxis = xON:xOFF;
yON = y; yOFF=y+h; VerticalAxis   = yON:yOFF;

% MouthZone=zeros(size(im1d,1),size(im1d,2)); MouthZone(VerticalAxis,HorizontalAxis)=1; %Here selects a square area
MouthZone= draw_elips_im(size(im1,2),size(im1,1),centroid(1),centroid(2),w/2,h/2,0);  % Here an ellipse

% (2) estimate lipcolor with K-means algorithm
r_threshold = 170; g_threshold = 70; b_threshold = 70;    
im2 = MouthZone.*(im1d(:,:,1)*r_threshold + im1d(:,:,2)*g_threshold + im1d(:,:,3)*b_threshold); % weight
im3 = MouthZone.*sqrt(im1d(:,:,1).^2 + im1d(:,:,2).^2 + im1d(:,:,3).^2);
co = im2./(im3+eps);
[a,b]=hist(co(co>0),100); % hist of colour distributions
metd='cityblock';
silM=zeros(5,1);
 %%% compute for up to K=5 clusters
 for K=2:5; silM(K)=mean(silhouette(b',kmeans(b,K, 'Distance',metd),metd)); end
[cl_idx, cl_center] = kmeans(b,argmax(silM),'distance',metd,'Replicates',7);
%[u, s, P] = fit_mix_gaussian_multiple(b',K); % consider using E.M. here
A=smooth(a); %smooth and select the cluster with more elements
for p=1:length(cl_center); AmplCluster(p) = A(argmin(abs(b-cl_center(p)))); end

LipColour = cl_center(argmax(AmplCluster));
% TeethClr  = cl_center(argmin(AmplCluster));
%% first estimate of ellipse parameters
fi = find(co >= LipColour );
% fi = find((co>LipColour*(1-.01) & co<LipColour*(1+.05)));
[a,b]   = sort(co(fi));
mask    = zeros(size(co));
mask(fi(b)) = 1;
tmp = imfilter(mask, fspecial('average',[5 5]) );
tmp(tmp<0.5) = 0; tmp(tmp>=0.5) = 1;
tmp = imclose(tmp,strel('disk',10));
B = bwconvhull_altern(tmp,'union');
    
    
st = regionprops(B,{'Area','MajorAxisLength','MinorAxisLength','Centroid','Orientation'});
st=st(1);    
fun=@(x) sum(sum(abs(draw_elips_im(size(B,2),size(B,1),x(1),x(2),x(3),x(4),x(5))-B))); %cost function
%     EllipVals = fminsearch(fun,[st.Centroid(1) st.Centroid(2) st.MajorAxisLength/2 st.MinorAxisLength/2]);
% Acon=[1 0 0 0 0; -1 0 0 0 0; 0 1 0 0 0; 0 -1 0 0 0; 0 0 1 0 0; 0 0 -1 0 0; 0 0 0 1 0; 0 0 0 -1 0; 0 0 0 0 1; 0 0 0 0 -1];
% bcon=[oldCx*(1+tol10); oldCx*(tol10-1); oldCy*(1+tol10); oldCy*(tol10-1); oldMAx*(1+tol10); oldMAx*(tol10-1); oldmAx*(1+tol10); oldmAx*(tol10-1);  oldOri*(1+tol10); oldOri*(tol10-1);];
% EllipVals = fminsearch(fun,[st.Centroid(1) st.Centroid(2) st.MajorAxisLength/2 st.MinorAxisLength/2 st.Orientation]);
% oldCx=EllipVals(1); oldCy=EllipVals(2); oldMAx=EllipVals(3); oldmAx=EllipVals(4); oldOri=EllipVals(4);
AreaIn=st.Area; MaxArea=(1+20/100)*st.Area; MinArea=(1-10/100)*st.Area;
%%
if flag_plot, f1=figure(1); set(f1, 'Position', get(0,'Screensize'));  end
LipStat=NaN(7,length(items));
for k=1:length(items)
    im1 = imread(strcat(pathV,'\',items(k).name));   
    im1d= double(im1(:,:,:));

    r_threshold = 170; g_threshold = 70; b_threshold = 70;    
    im2 = MouthZone.*(im1d(:,:,1)*r_threshold + im1d(:,:,2)*g_threshold + im1d(:,:,3)*b_threshold); % weight
    im3 = MouthZone.*sqrt(im1d(:,:,1).^2 + im1d(:,:,2).^2 + im1d(:,:,3).^2);
    co = im2./(im3+eps);
    
    fi = find(co >= LipColour );
    [a,b]   = sort(co(fi));
    mask    = zeros(size(co));
    mask(fi(b)) = 1;
    tmp = imfilter(mask,fspecial('average',[5 5])); %create averaging filter
    tmp(tmp < 0.5)  = 0;  tmp(tmp >= 0.5) = 1;
    tmp = imclose(tmp,strel('disk',10));
    B = bwconvhull_altern(tmp,'union');
    
%     comp = zeros(1,length(st)); 
%     for kk = 1:size_st, comp(kk) = st(kk).MajorAxisLength; end
%     maxval = max(comp);
%     maxind = find(comp == maxval);    
    st = regionprops(B,{'FilledImage','Area','MajorAxisLength','MinorAxisLength','Centroid','Orientation','Eccentricity','Extrema'});
    maxind=1; st=st(maxind);    
    tol10=20/100; tol05=15/100;
%     Acon=[1 0 0 0 0; -1 0 0 0 0; 0 1 0 0 0; 0 -1 0 0 0; 0 0 1 0 0; 0 0 -1 0 0; 0 0 0 1 0; 0 0 0 -1 0; 0 0 0 0 1; 0 0 0 0 -1];
%     bcon=[oldCx*(1+tol10); oldCx*(tol10-1); oldCy*(1+tol10); oldCy*(tol10-1); oldMAx*(1+tol10); oldMAx*(tol10-1); oldmAx*(1+tol05); oldmAx*(tol05-1);  oldOri*(1+tol05); oldOri*(tol05-1);];
%     EllipVals = fmincon(fun,[st.Centroid(1) st.Centroid(2) st.MajorAxisLength/2 st.MinorAxisLength/2 st.Orientation],Acon,bcon);
%     oldCx=EllipVals(1); 
%     oldCy=EllipVals(2);     
%     oldMAx=EllipVals(3);    
%     oldmAx=EllipVals(4);
%     oldOri=EllipVals(4);
%     B2=draw_elips_im(size(B,2),size(B,1),EllipVals(1),EllipVals(2),EllipVals(3),EllipVals(4),EllipVals(5));
%     [xx,yy]=find(((imdilate(B,strel('disk',1))-B))==1);    

if abs(st.Area-AreaIn)/AreaIn<40/100
    MouthZone = imdilate(B,strel('disk',8));
    AreaIn=st.Area;
    oldst=st;
else
    st=oldst;
end

    % Extract information about blob
    LipStat(1,k) = st(maxind).Area; % mouth area
    LipStat(2,k) = st(maxind).MajorAxisLength; % width of mouth opening
    LipStat(3,k) = st(maxind).MinorAxisLength; % height of mouth opening
    LipStat(4,k) = maxind; % tells you which field the info was extracted from - should be the same across all frames!
    LipStat(5,k) = mean2(imfilter(co.*B, fspecial('laplacian'), 'replicate', 'conv').^2); %extract color gradient
    LipStat(6,k) = st(maxind).Centroid(1);
    LipStat(7,k) = st(maxind).Centroid(2);

% Apply smoothness constraints
%     if ((st.Area<=MaxArea) && (st.Area>=MinArea)) ...
%             && ((st.Area)-AreaIn)>10/100%%pdist([centroids LipStat([6 7],k)]','euclidean')<10
%         MouthZone = imdilate(B,strel('disk',8));
%         AreaIn=st.Area;
%     end
%         MouthZone = imdilate(B,strel('disk',8));
        
%     display
    if flag_plot
        face(:,:,1)=double(im1(:,:,1)).*(1-(imdilate(MouthZone,strel('disk',4))-MouthZone));   
        face(:,:,2)=double(im1(:,:,2)).*(1-(imdilate(MouthZone,strel('disk',4))-MouthZone)); 
        face(:,:,3)=double(im1(:,:,3)).*(1-(imdilate(MouthZone,strel('disk',4))-MouthZone)); 
        subplot(221); imshow(uint8(face)); title('search area')
        face(:,:,1)=double(im1(:,:,1)).*(1-(imdilate(B,strel('disk',4))-B));   
        face(:,:,2)=double(im1(:,:,2)).*(1-(imdilate(B,strel('disk',4))-B)); 
        face(:,:,3)=double(im1(:,:,3)).*(1-(imdilate(B,strel('disk',4))-B)); 
        subplot(222); imshow(uint8(face)); title('seleced mouth area');%     ff1=figure(12); imshow(face_a); set(ff1,'position',[150 100 500 500]);
%         face(:,:,1)=double(im1(:,:,1)).*(1-(imdilate(B2,strel('disk',4))-B2));   
%         face(:,:,2)=double(im1(:,:,2)).*(1-(imdilate(B2,strel('disk',4))-B2)); 
%         face(:,:,3)=double(im1(:,:,3)).*(1-(imdilate(B2,strel('disk',4))-B2)); 
%         subplot(223); imshow(uint8(face)); title('seleced mouth area');%     ff1=figure(12); imshow(face_a); set(ff1,'position',[150 100 500 500]);
        pause(1e-4);
        set(gcf, 'Position', get(0,'Screensize')); 
        subplot(224); imagesc(co); axis([200 350 150 250]);
%         pause
    end
    
    if k/4==fix(k/4), fprintf(1,'.'); end
  
end
%% display
if flag_plot
    ym=normaNr(LipStat(1,:),min(Rms),max(Rms)); yr=smooth(Rms)';
    figure(31); plot(t,ym,'-k.',t,yr,'r-',t,0.25*(abs(ym-yr)),'--b','linewidth',3); 
    legend('m-SNR','s-SNR',['r=',num2str(corr(ym',yr'),2),' ;e=',num2str(rmse(ym,yr),2)],'location','best');
end

output = struct;
output.LipStat   = LipStat;
output.RMS       = Rms;
output.x         = sig;
output.fs        = fs;
output.LipColour = LipColour;
output.pathV     = pathV;
output.t         = t;
output.startArea = [xON xOFF yON yOFF];


end %EoF

function P = bwconvhull_alt(BW)

% usage: 
% - BW is the input binary image
% - P is a binary image wherein the convex hull of objects are returned
% P = bwconvhull_alt(BW);

warning off all
s=regionprops(logical(BW),'ConvexImage','BoundingBox');
P=zeros(size(BW));
for no=1:length(s)
    P(s(no).BoundingBox(2):s(no).BoundingBox(2)+s(no).BoundingBox(4)-1,...
        s(no).BoundingBox:s(no).BoundingBox(1)+s(no).BoundingBox(3)-1)=s(no).ConvexImage;
end

end

function convex_hull = bwconvhull_altern(varargin)
%BWCONVhull Generate convex hull image from binary image.
%   CH = BWCONVHULL(BW) computes the convex hull of all objects in BW and
%   returns CH, binary convex hull image.  BW is a logical 2D image and CH
%   is a logical convex hull image, containing the binary mask of the
%   convex hull of all foreground objects in BW.
%
%   CH = BWCONVHULL(BW,METHOD) specifies the desired method for computing
%   the convex hull image.  METHOD is a string and may have the following
%   values:
%
%      'union'   : Compute convex hull of all foreground objects, treating
%                  them as a single object.  This is the default method.
%      'objects' : Compute the convex hull of each connected component of
%                  BW individually.  CH will contain the convex hulls of
%                  each connected component.
%
%   CH = BWCONVHULL(BW,'objects',CONN) specifies the desired connectivity
%   used when defining individual foreground objects.  The CONN parameter
%   is only valid when the METHOD is 'objects'.  CONN may have the
%   following scalar values:
%
%      4 : two-dimensional four-connected neighborhood
%      8 : two-dimensional eight-connected neighborhood {default}
%
%   Additionally, CONN may be defined in a more general way, using a 3-by-3
%   matrix of 0s and 1s.  The 1-valued elements define neighborhood
%   locations relative to the center element of CONN.  CONN must be
%   symmetric about its center element.
%
%   Example
%   -------
%   subplot(2,2,1);
%   I = imread('coins.png');
%   imshow(I);
%   title('Original');
%   
%   subplot(2,2,2);
%   BW = I > 100;
%   imshow(BW);
%   title('Binary');
%   
%   subplot(2,2,3);
%   CH = bwconvhull(BW);
%   imshow(CH);
%   title('Union Convex Hull');
%   
%   subplot(2,2,4);
%   CH_objects = bwconvhull(BW,'objects');
%   imshow(CH_objects);
%   title('Objects Convex Hull');
%
%   See also BWCONNCOMP, BWLABEL, LABELMATRIX, REGIONPROPS.

%   Copyright 2010 The MathWorks, Inc.
%   $Revision: 1.1.6.7 $  $Date: 2011/08/09 17:49:02 $

[BW, method, conn] = parseInputs(varargin{:});

% Label the image
if strcmpi(method,'union')
    % 'union' : label all 'true' pixels as a single region
    labeled_image = uint8(BW);
else
    % 'objects' : label as normal
    labeled_image = bwconncomp(BW,conn);
end

% Call regionprops
blob_props = regionprops(labeled_image,'BoundingBox','ConvexImage');
num_blobs = length(blob_props);
[rows columns] = size(BW);

% Loop over all blobs getting the CH for each blob one at a time, then add
% it to the cumulative CH image.
convex_hull = false(rows, columns);
for i = 1 : num_blobs
    m = blob_props(i).BoundingBox(4);
    n = blob_props(i).BoundingBox(3);
    r1 = blob_props(i).BoundingBox(2) + 0.5;
    c1 = blob_props(i).BoundingBox(1) + 0.5;
    rows = (1:m) + r1 - 1;
    cols = (1:n) + c1 - 1;
    convex_hull(rows,cols) = convex_hull(rows,cols) | blob_props(i).ConvexImage;
end


%------------------------------------------------
function [BW,method,conn] = parseInputs(varargin)

%narginchk(1,3);

BW = varargin{1};
validateattributes(BW, {'logical' 'numeric'}, {'2d', 'real', 'nonsparse'}, ...
    mfilename, 'BW', 1);

if ~islogical(BW)
    BW = BW ~= 0;
end

if nargin == 1
    % BWCONVHULL(BW)
    method = 'union';
    conn = 8;
    
elseif nargin == 2
    % BWCONVHULL(BW,METHOD)
    method = varargin{2};
    conn = 8;
    
else
    % BWCONVHULL(BW,METHOD,CONN)
    method = varargin{2};
    conn = varargin{3};
    
    % special case so that we go through the 2D code path for 4 or 8
    % connectivity
    if isequal(conn, [0 1 0;1 1 1;0 1 0])
        conn = 4;
    end
    if isequal(conn, ones(3))
        conn = 8;
    end
    
end

% validate inputs (accepts partial string matches)
method = validatestring(method,{'union','objects'},mfilename,'METHOD',2);

% validate connectivity
is_valid_scalar = isscalar(conn) && (conn == 4 || conn == 8);
if is_valid_scalar
    return
end

% else, validate 3x3 connectivity matrix

% 3x3 matrix...
is_valid_matrix = isnumeric(conn) && isequal(size(conn),[3 3]);
% with all 1's and 0's...
is_valid_matrix = is_valid_matrix && all((conn(:) == 1) | (conn(:) == 0));
% whos center value is non-zero
is_valid_matrix = is_valid_matrix && conn((end+1)/2) ~= 0;
% and which is symmetrix
is_valid_matrix = is_valid_matrix && isequal(conn(1:end), conn(end:-1:1));

if ~is_valid_matrix
    error(message('images:bwconvhull:invalidConnectivity'))
end
end
end


function [u,sig,P,t,iter] = fit_mix_gaussian_multiple( X, K )
%% fits K gaussians assuming multivariate distrib 
% implements: EM algorithm. 
% input:    X   - input samples, Nx1 vector
%           K   - number of gaussians which are assumed to compose the distribution
%
% output:   u   - fitted mean for each gaussian
%           sig - fitted standard deviation for each gaussian
%           t   - probability of each gaussian in the complete distribution
%           iter- number of iterations done by the function

% initialize and initial guesses
N           = length( X );
Z           = ones(N,K) * 1/K;                  % indicators vector
t           = ones(1,K) * 1/K;                  % distribution of the gaussian models in the samples
u           = linspace(min(X),max(X),K);        % mean vector
sig2        = ones(1,K) * var(X) / sqrt(K);     % variance vector
C           = 1/sqrt(2*pi);                     % just a constant
Ic          = ones(N,1);                        % - enable a row replication by the * operator
Ir          = ones(1,K);                        % - enable a column replication by the * operator
thresh      = 1e-3;
step        = N;
last_step   = inf;
iter        = 0;
min_iter    = 10;
max_iter    = 1e3;

% main convergence loop, assume gaussians are 1D
while ((( abs((step/last_step)-1) > thresh) && (step>(N*eps)) && iter<max_iter ) || (iter<min_iter) ) 
    
    % E step
    % ========
    Q   = Z;
    P   = C ./ (Ic*sqrt(sig2)) .* exp( -((X*Ir - Ic*u).^2)./(2*Ic*sig2) );
    for m = 1:K
        Z(:,m)  = (P(:,m)*t(m))./(P*t(:));
    end
        
    % estimate convergence step size and update iteration number
%     prog_text   = sprintf(repmat( '\b',1,(iter>0)*12+ceil(log10(iter+1)) ));
    iter        = iter + 1;
    last_step   = step * (1 + eps) + eps;
    step        = sum(sum(abs(Q-Z)));
%     fprintf( '%s%d iterations\n',prog_text,iter );

    % M step
    % ========
    Zm        = sum(Z);               % sum each column
    Zm(Zm==0) = eps;                  % avoid devision by zero
    u         = (X')*Z ./ Zm;
    sig2      = sum(((X*Ir - Ic*u).^2).*Z) ./ Zm;
    t         = Zm/N;
end

sig     = sqrt( sig2 );
end

function txt = myupdatefcn(empty, event_obj)
  global ru
  txt = {''};
  ru(end+1,:)=event_obj.Position;
  ru=unique(ru,'rows');
  if iseven(size(ru,1))
      x = min(ru(end-1,1),ru(end,1)); y = min(ru(end-1,2),ru(end,2));
      w = abs(ru(end-1,1)-ru(end,1)); h = abs(ru(end-1,2)-ru(end,2)); 
      hold on;
      rectangle('Position',[x y w h],'EdgeColor','r','Tag', 'myRect'); 
%       txt = {'close window if OK, otherwise re-measure'};
  elseif (isodd(size(ru,1)) && size(ru,1)>1)
      ru=[];
      delete(findobj(allchild(gcf), 'Tag', 'myRect'))
  end
end

function [Y]=normaNr(x,m,M)
if size(x,2)==1
    Y=(x-min(x))/(max(x)-min(x))*(M-m)+m;
else
    X_m=min(min(x));
    X_M=max(max(x));
    Y=(x-X_m)/(X_M-X_m)*(M-m)+m;
end
end %EoF

function x = iseven(number)
a = number/2;
whole = floor(a);
part = a-whole;
if part>0, x=1;    
else       x=0;   
end
x=~x;
end
function x = isodd(number)
x=~iseven(number);    
end

function ellipsePixels = draw_elips_im(imageSizeX,imageSizeY,centerX,centerY,radiusX,radiusY,Orient)
 %%
% imageSizeX = 640; imageSizeY = 480;
[columnsInImage rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
% Next create the ellipse in the image.
% centerX = 320; centerY = 240; radiusX = 250; radiusY = 150;
ellipsePixels = ...
    (rowsInImage - centerY).^2 ./ radiusY^2 + (columnsInImage - centerX).^2 ./ radiusX^2 <= 1;

ellipsePixels = imrotate(ellipsePixels,Orient,'crop');
end