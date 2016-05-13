function output = lipTracker(filepath,flag_plot,flag_manual)
%%% script to perform computation of mouth area
%%% [SC] May-16
% This script performs a computation of the mouth area (including lips)
% using mostly hue. The algorithm performs a computation of the colour
% clusters (mostly skin and lip) using k-means algorithm. The standard
% deviation of these clusters is estimated by fitting up to 5 gaussian
% distributions to the histogram of the color leveles (optimization done
% with E.M.). To improve the algorithm, some smoothness is applied
% between frames, so that the area cannot vary >60% between consecutive
% frames, and the "search area" for the lips also varies accordingly. 
% The user needs to initiate the algorithm by selecting the search area in
% the first frame using the prompted GUI.

% This algorithm was tested ONLY with the VidTIMIT database. If your videos
% have different format, you'll probably need to modify the code accordingly. 

% input:   
% - filepath. This refers to a location that containts both video and
%   audio. For VidTIMIT, frames and audio are in separate folders. 
%   (See example below.)
% - flag_plot. Plot frame-by-frame the results of lip detection/traking.
% - flag_manual. When the algorithm fails, run with flag_manual=true to
%   check that each frame is labelled correctly and, if necessary, correct 
%   the mouth area.

% With flag_manual=true, the user will be prompted to check that the
% correct mouth area has been selected. If the user intends to select a
% different area, the user needs to click twice in the subplot 2.24. 
% This will display a red square and the algorithm will re-run to detect
% the mouth for that frame. 
% REMEMBER to always press OK after after selecting the new mouth-area, or 
% if you're already happy with estimated area.

% Example: How to run this script.
% 
% clear all; clc; close all;
% DBFILEPATH='C:\MATLAB\VidTIMIT\';
% talkers = dir(DBFILEPATH); talkers = talkers(3:end);
% l_TALKERS=length(talkers); %no. of talkers
% 
% for tt=1:l_TALKERS
%     subj  = talkers(tt).name;
%     sents = dir(strcat(DBFILEPATH, subj,'\video')); sents = sents(3:end);
%     l_SENTS = length(sents);
%     for ss=1:l_SENTS%ss=4
%         sent=sents(ss).name;
%         if ~any([strcmp(sent,'head') strcmp(sent,'head2') strcmp(sent,'head3')]) 
%         % exclude folders 'head', 'head2' and 'head3'.
%         filepath=strcat(DBFILEPATH,subj,'\video\',sent);
%         fprintf(1,'\nTalker: %s (%d of %d)\tSent: %s (%d of %d)',subj,tt,l_TALKERS,sent,ss,l_SENTS);
%         output = lipTracker(filepath, true, true);
% %         save(strcat('MTk_',subj,'_',sent), 'output');
%         end
%     end
% end

if nargin<3, flag_manual=false;
    if nargin<2, flag_plot=false; 
end;end;
if flag_manual, flag_plot=true; end 

pathV=filepath;
pathA=[strrep(pathV,'video','audio'),'.wav'];

items = dir(pathV); items = items(3:end);
l_f=length(items); %no. of frames

% (0) Get audio RMS
[sig,fs]=wavread(pathA); sig=sig./max(sig);
N=length(sig)/l_f;
for n=1:l_f, Rms(n) = rms(sig(1+(n-1)*N:n*N)); end
t=linspace(0,length(sig)/fs,l_f);

% % (1) Manually select mouth area
% Selecting the two vartices of a rectangle will select the mouth area
% (Note that the actual area that is used is an ellipse)
% If the first selection is not accurate, repeat the process without 
% closing the figure. A new red square appear every two clicks.
close(figure(1));
im1 = imread(strcat(pathV,'\',items(1).name));   
im1d= double(im1(:,:,:));
f1=figure(1); 
imshow(im1); title 'SELECT MOUTH AREA with mouse-clicks';
set(f1, 'Position', get(0,'Screensize')); 
global ru; ru=[];
dcm_obj = datacursormode(gcf);
datacursormode on;
set(dcm_obj,'UpdateFcn', @myupdatefcn)
waitfor(f1);

x = min(ru(end-1,1),ru(end,1)); y = min(ru(end-1,2),ru(end,2));
w = abs(ru(end-1,1)-ru(end,1)); h = abs(ru(end-1,2)-ru(end,2)); 

% x =230;y=203;w=58;h=19; %for testing with tt=1, ss=4;
% x=220;y=213;w=69;h=34; %for testing with tt=2, ss=9;

centroid=[x+w/2; y+h/2];     
xON = x; xOFF=x+w; %HorizontalAxis = xON:xOFF;
yON = y; yOFF=y+h; %VerticalAxis   = yON:yOFF;
% MouthZoneSqrd=zeros(size(im1d,1),size(im1d,2)); MouthZone(VerticalAxis,HorizontalAxis)=1; %Here selects a square area
MouthZone= draw_elips_im(size(im1,2),size(im1,1),centroid(1),centroid(2),w/2,h/2,0);        % Here an ellipse (zero orientation)

% (2) create colour plane for mouth tracking (not sure about these values)
r_threshold = 170; g_threshold = 70; b_threshold = 70;    
im2 = MouthZone.*(im1d(:,:,1)*r_threshold + im1d(:,:,2)*g_threshold + im1d(:,:,3)*b_threshold); % weight
im3 = MouthZone.*sqrt(im1d(:,:,1).^2 + im1d(:,:,2).^2 + im1d(:,:,3).^2);
co  = im2./(im3+eps);

% (3) estimate Lipcolor/s with K-means algorithm + E.M.
[LipColour, SdLipColr, SkinColor]=est_lip_color(co);

% Find mouth area for frame 1
fi = find(co >= LipColour-SdLipColr );
[a,b]   = sort(co(fi));
mask    = zeros(size(co));
mask(fi(b)) = 1;
tmp = imclose(mask,strel('disk',20));    
B = bwconvhull_altern(tmp,'union');
      
st = regionprops(B,{'Area','MajorAxisLength','MinorAxisLength','Centroid','Orientation'});
st=st(1);    
% Optimize relative to an ellipse
% fun=@(x) sum(sum(abs(draw_elips_im(size(B,2),size(B,1),x(1),x(2),x(3),x(4),x(5))-B))); %cost function
% %     EllipVals = fminsearch(fun,[st.Centroid(1) st.Centroid(2) st.MajorAxisLength/2 st.MinorAxisLength/2]);
% % Acon=[1 0 0 0 0; -1 0 0 0 0; 0 1 0 0 0; 0 -1 0 0 0; 0 0 1 0 0; 0 0 -1 0 0; 0 0 0 1 0; 0 0 0 -1 0; 0 0 0 0 1; 0 0 0 0 -1];
% % bcon=[oldCx*(1+tol10); oldCx*(tol10-1); oldCy*(1+tol10); oldCy*(tol10-1); oldMAx*(1+tol10); oldMAx*(tol10-1); oldmAx*(1+tol10); oldmAx*(tol10-1);  oldOri*(1+tol10); oldOri*(tol10-1);];
% EllipVals = fminsearch(fun,[st.Centroid(1) st.Centroid(2) st.MajorAxisLength/2 st.MinorAxisLength/2 st.Orientation]);
% oldCx=EllipVals(1); oldCy=EllipVals(2); oldMAx=EllipVals(3); oldmAx=EllipVals(4); oldOri=EllipVals(4);

% Area0=AreaIn; MaxArea=(1+20/100)*st.Area; MinArea=(1-10/100)*st.Area;
AreaIn=st.Area;  MZold=MouthZone; 

if flag_plot, f1=figure(1); set(f1, 'Position', get(0,'Screensize'));  end
LipStat=NaN(7,length(items)); %initialize

k = 1; %start from frame number
while k<=length(items)
    
    % (0) load frame
    im1 = imread(strcat(pathV,'\',items(k).name));   
    im1d= double(im1(:,:,:));

    % (1) create colour plane 
    r_threshold = 170; g_threshold = 70; b_threshold = 70;    
    im2 = MouthZone.*(im1d(:,:,1)*r_threshold + im1d(:,:,2)*g_threshold + im1d(:,:,3)*b_threshold); % weight
    im3 = MouthZone.*sqrt(im1d(:,:,1).^2 + im1d(:,:,2).^2 + im1d(:,:,3).^2);
    co = im2./(im3+eps);
    
    % (2) move colour levels != from lip to the skin cluster
    co2=co; co2(co2<LipColour-2.5*SdLipColr & co2>0)=SkinColor+.1*randn(1,length(co2(co2<LipColour-2.5*SdLipColr & co2>0))); co=co2; clear co2;
    
    % (3) re-estimate lip colour
    [LipColour, SdLipColr]=est_lip_color(co,LipColour,SdLipColr);
    if flag_plot, fprintf(1,'\n%d:\tLipColour%.1f; SdLipColr: %.1f',k,LipColour,SdLipColr); end
    

    % (4) filter clusters
    %     fi = find(co>=LipColour-SdLipColr & co<=LipColour+2*SdLipColr);
    fi = find(co>=LipColour-SdLipColr);
    [a,b]   = sort(co(fi));
    mask    = zeros(size(co));
    mask(fi(b)) = 1;
    %    mask=smooth_expin(mask,4,4); %additional smoother, not necessary
    tmp = imclose(mask,strel('disk',20));    
    
    % (5) binarize and get shapes
    B = bwconvhull_altern(tmp,'union');    
    st = regionprops(B,{'FilledImage','Area','MajorAxisLength','MinorAxisLength','Centroid','Orientation','Eccentricity','Extrema'});
    maxind=1; st=st(maxind);    

%     % Alterntively, fit an ellipse
%     tol10=20/100; tol05=15/100;
%     Acon=[1 0 0 0 0; -1 0 0 0 0; 0 1 0 0 0; 0 -1 0 0 0; 0 0 1 0 0; 0 0 -1 0 0; 0 0 0 1 0; 0 0 0 -1 0; 0 0 0 0 1; 0 0 0 0 -1];
%     bcon=[oldCx*(1+tol10); oldCx*(tol10-1); oldCy*(1+tol10); oldCy*(tol10-1); oldMAx*(1+tol10); oldMAx*(tol10-1); oldmAx*(1+tol05); oldmAx*(tol05-1);  oldOri*(1+tol05); oldOri*(tol05-1);];
%     EllipVals = fmincon(fun,[st.Centroid(1) st.Centroid(2) st.MajorAxisLength/2 st.MinorAxisLength/2 st.Orientation],Acon,bcon);
%     oldCx=EllipVals(1); oldCy=EllipVals(2); oldMAx=EllipVals(3); oldmAx=EllipVals(4); oldOri=EllipVals(4);
%     B2=draw_elips_im(size(B,2),size(B,1),EllipVals(1),EllipVals(2),EllipVals(3),EllipVals(4),EllipVals(5));
    
    % (6) Apply smoothness constraints
    TolAPr=60/100;
    AreaDiff = (st.Area-AreaIn)/AreaIn;
    %if the area computed for this frame is too different from that of previous
    %frame, discard computation. This is over-ruled when running the script
    %with flag_manual=true (see below).
    if abs(AreaDiff)>TolAPr, st=oldst; fprintf(1,' - NOT COUNT');
    else    MouthZone= imdilate(draw_elips_im(size(im1,2),size(im1,1),...
                max(centroid(1)-15,min(centroid(1)+15,st.Centroid(1))),...
                max(centroid(2)-15,min(centroid(2)+15,st.Centroid(2))),...
                w/2,2*h/3,max(-8,min(+8,st.Orientation))),strel('disk',2));
            oldst=st;
    end

    % (6.b) consider re-estimating the frame
    if flag_manual && k>1
        % for frames>1, check that the selected mouth area is correct
        % The user can use mouse-clicks to re-run the lip tracking algorithm
        % for that frame. Otherwise, simply click OK.
        
        uicontrol('Style', 'pushbutton', 'String', {'OK' 'next frame'},...
                       'Position', [870 480 150 30],...
                       'Callback', 'uiresume(gcbf)');
        drawnow();                   


        % (6.b.1) show search area, selected area and open GUI to modify selection
        face(:,:,1)=double(im1(:,:,1)).*(1-(imdilate(MZold,strel('disk',3))-MZold)-1.*(imdilate(B,strel('disk',3))-B));   
        face(:,:,2)=double(im1(:,:,2)).*(1-(imdilate(MZold,strel('disk',3))-MZold)-0.*(imdilate(B,strel('disk',3))-B)); 
        face(:,:,3)=double(im1(:,:,3)).*(1-(imdilate(MZold,strel('disk',3))-MZold)-0.*(imdilate(B,strel('disk',3))-B)); 
        subplot(224); hold off; imshow(uint8(face)); axis([xON-10 xOFF+10 yON-15 yOFF+15]); title(['Mouth selection for next frame (#', num2str(k) ,'). OK/Re-select area']);

        % (6.b.2) get new mouth area from the user
        ru=[]; 
        dcm_obj = datacursormode(gcf); datacursormode on;
        set(dcm_obj,'UpdateFcn', @myupdatefcn)
        uiwait(f1); 
        if ~isempty(ru) %if the user inputs a new mouth area
        if iseven(size(ru,1))
        x = min(ru(end-1,1),ru(end,1)); y = min(ru(end-1,2),ru(end,2));
        w = abs(ru(end-1,1)-ru(end,1)); h = abs(ru(end-1,2)-ru(end,2)); 
        centroid=[x+w/2; y+h/2];     
        MouthZone= draw_elips_im(size(im1,2),size(im1,1),centroid(1),centroid(2),w/2,h/2,0);        % Here an ellipse (zero orientation)
        
        % (6.b.3) 
        k=k-1; % reset index to re-compute the mouth area for this frame
        end
        end
    end
        
    % Save info 
    LipStat(1,k) = st.Area; % mouth area
    LipStat(2,k) = st.MajorAxisLength; % width of mouth opening
    LipStat(3,k) = st.MinorAxisLength; % height of mouth opening
    LipStat(4,k) = LipColour; % lip colour for this frame
    LipStat(5,k) = mean2(imfilter(co.*B, fspecial('laplacian'), 'replicate', 'conv').^2); %extract color gradient
    LipStat(6,k) = st(maxind).Centroid(1);
    LipStat(7,k) = st(maxind).Centroid(2);

    % Display info
    if flag_plot 
        face(:,:,1)=double(im1(:,:,1)).*(1-(imdilate(MZold,strel('disk',3))-MZold)-1.*(imdilate(B,strel('disk',3))-B));   
        face(:,:,2)=double(im1(:,:,2)).*(1-(imdilate(MZold,strel('disk',3))-MZold)-0.*(imdilate(B,strel('disk',3))-B)); 
        face(:,:,3)=double(im1(:,:,3)).*(1-(imdilate(MZold,strel('disk',3))-MZold)-0.*(imdilate(B,strel('disk',3))-B)); 
        subplot(222); imshow(uint8(face)); title(['seleced mouth area; frame: ',num2str(k)]);%     ff1=figure(12); imshow(face_a); set(ff1,'position',[150 100 500 500]);
        subplot(221); histn(co(co>0),100); hold on; plot([LipColour LipColour],[0 1],'r',[LipColour-SdLipColr LipColour-SdLipColr],[0 1],'--r',[LipColour+SdLipColr LipColour+SdLipColr],[0 1],'--r'); axis([SkinColor-8 LipColour+8 0 1.05]); hold off; title 'colour hist';
        subplot(223); imagesc(co<LipColour); axis([xON-20 xOFF+20 yON-20 yOFF+20]); title 'only lip pixels';
        if ~flag_manual
            subplot(224); imagesc(co); axis([xON-20 xOFF+20 yON-20 yOFF+20]); title 'selected on the colour plane';'';
        end
        drawnow()
        set(gcf, 'Position', get(0,'Screensize'));    
        MZold=MouthZone; % save search area       
%         pause
    end
    
    if k/4==fix(k/4), fprintf(1,'.'); end %display '.' to screen

    k=k+1;
end
%% final plot with
if flag_plot 
    ym=normaNr(LipStat(1,:),min(Rms),max(Rms)); yr=smooth(Rms)';
    figure(31); plot(t,ym,'-k.',t,yr,'r-','linewidth',3); 
    legend('m-SNR','s-SNR','location','best'); title(['r=',num2str(corr(ym',yr'),2),' ;e=',num2str(rmse(ym,yr),2)]);    
end

% output data
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

function [LipColour, SdLipColr, SkinColor]=est_lip_color(co,prevLC,prevSD)

[a,b]=hist(co(co>0),100); % hist of colour distributions
metd='cityblock';
silM=zeros(5,1);
 %%% compute for up to K=5 clusters
 for K=2:5; silM(K)=mean(silhouette(b',kmeans(b,K, 'Distance',metd),metd)); end

 %%% find clusters
 [cl_idx, cl_center] = kmeans(b,argmax(silM),'distance',metd,'Replicates',7);

A=smooth(a); %smooth and select the cluster with more elements
for p=1:length(cl_center); AmplCluster(p) = A(argmin(abs(b-cl_center(p)))); end

% use apriori about lip color being "redder" than skin
LipColour = max(cl_center);% LipColour = cl_center(argmax(AmplCluster));
SkinColor = min(cl_center);% SkinColor = cl_center(argmin(AmplCluster));

%find std of guassians
[u, s] = fit_mix_gaussian_multiple(b',length(cl_center)); % E.M.
SdLipColr = s(argmin(abs(u-LipColour)));

%average with previous values
if nargin==3,   LipColour = (LipColour + prevLC)/2;
                SdLipColr = (SdLipColr + prevSD)/2;    
end

end %EoF
% function P = bwconvhull_alt(BW)
% 
% % usage: 
% % - BW is the input binary image
% % - P is a binary image wherein the convex hull of objects are returned
% % P = bwconvhull_alt(BW);
% 
% warning off all
% s=regionprops(logical(BW),'ConvexImage','BoundingBox');
% P=zeros(size(BW));
% for no=1:length(s)
%     P(s(no).BoundingBox(2):s(no).BoundingBox(2)+s(no).BoundingBox(4)-1,...
%         s(no).BoundingBox:s(no).BoundingBox(1)+s(no).BoundingBox(3)-1)=s(no).ConvexImage;
% end
% 
% end%EoF

function histn(D,N)
% custom normalized hist
[h, a] = hist(D,N); %N= 100 bins.
h = h/sum(h); % normalize to unit length. Sum of h now will be 1.
h = norma(h);
bar(a,h); 
end   
function convex_hull = bwconvhull_altern(varargin)


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
end%EoF


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
end%EoF

function txt = myupdatefcn(empty, event_obj)
  % draws a red square in the image every two button-clicks
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
end%EoF

function [Y]=normaNr(x,m,M)
% normalise the input
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
end%EoF
function x = isodd(number)
x=~iseven(number);    
end%EoF

function ellipsePixels = draw_elips_im(imageSizeX,imageSizeY,centerX,centerY,radiusX,radiusY,Orient)
% draws an elliupse
if nargin==3
    tmp=centerX;
    centerX   =tmp(1);
    centerY   =tmp(2);
    radiusX   =tmp(3);
    radiusY   =tmp(4);
    Orient    =tmp(5);
end
% imageSizeX = 640; imageSizeY = 480;
[columnsInImage rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
% Next create the ellipse in the image.
% centerX = 320; centerY = 240; radiusX = 250; radiusY = 150;
ellipsePixels = ...
    (rowsInImage - centerY).^2 ./ radiusY^2 + (columnsInImage - centerX).^2 ./ radiusX^2 <= 1;

ellipsePixels = imrotate(ellipsePixels,Orient,'crop');
end %EoF

% function y=Nhside(x)
% % negative heaviside function
% y=0; if x>0, y=1; elseif x<0, y=-1; end
% end %EoF