%%% CamKas
%%% script to aid labelling of mouth status
%%%
%%% [SC] March-16
clear all; clc;
subj='fadg0'; sent='sa1';
pathV=strcat('C:\Users\sc04\Documents\MATLAB\',subj,'\video\',sent);
pathA=strcat('C:\Users\sc04\Documents\MATLAB\',subj,'\audio\',sent,'.wav');
items = dir(pathV); items = items(3:end);
l_f=length(items); %no. of frames
% process audio
[x,fs]=wavread(pathA); x=norma(x);
N=length(x)/l_f;
for n=1:l_f Rms(n) = rms(x(1+(n-1)*N:n*N)); end
t=linspace(0,length(x)/fs,l_f);
%% process video
% rep=1; % number of repetitions
% R=zeros(length(items)-2,rep);
% %% GUI to label manually m-SNR
% figure(1);
% for r=1:rep
%     for i=1:length(items)
% %         if length(items(i).name)>2
%             subplot(5,3,1:3)
%             plot(1:length(x),x,'k',1+(i-1)*N:i*N,x(1+(i-1)*N:i*N),'r');
%             xlim([1+(i-1)*N-4*N i*N+4*N]);
%             I = imread(strcat(pathV,'\',items(i).name));
%             subplot(5,3,4:15)
%             imshow(I);
%             tmp=input('open / kinda / closed'); if isempty(tmp); tmp=3; end
%             Rr(i,r)=tmp;
% %         end
%     end
%     pause(2);
% end
%save(strcat('R-',subj,'-',sent),'R');

%% compute mouth area using lip-tracking
% 
% % first select area
% clc; close all;
% im1 = imread(strcat(pathV,'\',items(1).name));   
% im1d= double(im1(:,:,:));
% f1=figure(1); imshow(im1); title 'SELECT MOUTH AREA'
% global ru; ru=[];
% dcm_obj = datacursormode(gcf);
% datacursormode on;
% set(dcm_obj,'UpdateFcn', @myupdatefcn)%, 'KeyPressFcn',@myupdatePdwnfcn
% waitfor(f1);
% x = min(ru(end-1,1),ru(end,1)); y = min(ru(end-1,2),ru(end,2));
% w = abs(ru(end-1,1)-ru(end,1)); h = abs(ru(end-1,2)-ru(end,2)); 
% centroids=[x+w/2; y+h/2];     
% 
% %
% xON = x; xOFF=x+w; HorizontalAxis = xON:xOFF;% xON = 200; xOFF=300; HorizontalAxis = xON:xOFF;
% yON = y; yOFF=y+h; VerticalAxis   = yON:yOFF; % yON = 180; yOFF=250; VerticalAxis   = yON:yOFF; 
% MouthZone=zeros(size(im1d,1),size(im1d,2));
% MouthZone(VerticalAxis,HorizontalAxis)=1;
%%
clear all; clc; close all; load('lip_data');
f1=figure(1); set(f1,'position',[150 100 900 500]);  

% estimate lipcolor

r_threshold = 170; g_threshold = 70; b_threshold = 70;    
im2 = MouthZone.*(im1d(:,:,1)*r_threshold + im1d(:,:,2)*g_threshold + im1d(:,:,3)*b_threshold); % weight
im3 = MouthZone.*sqrt(im1d(:,:,1).^2 + im1d(:,:,2).^2 + im1d(:,:,3).^2);
co = im2./(im3+eps);
[a,b]=hist(co(co>0),100);
[u,sig,tdfj,iter] = fit_mix_gaussian_multiple(b',1);
LipColour = max(u);

LipStat=NaN(7,length(items));
for k=1:length(items)
    im1 = imread(strcat(pathV,'\',items(k).name));   
    im1d= double(im1(:,:,:));

    r_threshold = 170; g_threshold = 70; b_threshold = 70;    
    im2 = MouthZone.*(im1d(:,:,1)*r_threshold + im1d(:,:,2)*g_threshold + im1d(:,:,3)*b_threshold); % weight
    im3 = MouthZone.*sqrt(im1d(:,:,1).^2 + im1d(:,:,2).^2 + im1d(:,:,3).^2);
    co = im2./(im3+eps);
    
    fi = find(co > LipColour);
    [a,b]   = sort(co(fi));
    mask    = zeros(size(co));
    mask(fi(b)) = 1;
    h       = fspecial('average',[5 5]); % create averaging filter
    se      = strel('disk',10);
    tmp = imfilter(mask,h);
    tmp(tmp < 0.5)  = 0;
    tmp(tmp >= 0.5) = 1;
    tmp = imclose(tmp,se);
    B = bwconvhull_alt(tmp);
    %100*sum(B(:)==1)/numel(B), imagesc(B);
    
    st = regionprops(B,{'Area','MajorAxisLength','MinorAxisLength','Centroid'});
    size_st = length(st);
    
    % Use e.g. MajorAxisLength to find the biggest blob (= the lips). 
    comp = zeros(1,length(st)); 
    for x = 1:size_st, comp(x) = st(x).MajorAxisLength; end
    maxval = max(comp);
    maxind = find(comp == maxval);
    
    % Extract information about the largest blob (using maxind)
    LipStat(1,k) = st(maxind).Area; % mouth area
    LipStat(2,k) = st(maxind).MajorAxisLength; % width of mouth opening
    LipStat(3,k) = st(maxind).MinorAxisLength; % height of mouth opening
    LipStat(4,k) = maxind; % tells you which field the info was extracted from - should be the same across all frames!
    LipStat(5,k) = mean2(imfilter(co.*B, fspecial('laplacian'), 'replicate', 'conv').^2); %extract color gradient
    LipStat(6,k) =  st(maxind).Centroid(1);
    LipStat(7,k) =  st(maxind).Centroid(2);

% Apply smoothness constraints
    if pdist([centroids LipStat([6 7],k)]','euclidean')<10
        MouthZone = imdilate(B,strel('disk',8));
    end
        
    % display
    face(:,:,1)=double(im1(:,:,1)).*(1-(imdilate(MouthZone,strel('disk',4))-MouthZone));   
    face(:,:,2)=double(im1(:,:,2)).*(1-(imdilate(MouthZone,strel('disk',4))-MouthZone)); 
    face(:,:,3)=double(im1(:,:,3)).*(1-(imdilate(MouthZone,strel('disk',4))-MouthZone)); 
    subplot(121); imshow(uint8(face)); title('search area')
    face(:,:,1)=double(im1(:,:,1)).*(1-(imdilate(B,strel('disk',4))-B));   
    face(:,:,2)=double(im1(:,:,2)).*(1-(imdilate(B,strel('disk',4))-B)); 
    face(:,:,3)=double(im1(:,:,3)).*(1-(imdilate(B,strel('disk',4))-B)); 
    subplot(122); imshow(uint8(face)); title('seleced mouth area');%     ff1=figure(12); imshow(face_a); set(ff1,'position',[150 100 500 500]);
    pause(1e-4);
  
end
ma=LipStat(1,:)-min(LipStat(1,:));
%%
figure(30)
ym=normaNr(ma,min(Rms),max(Rms)); yr=smooth(Rms)';
figure(31); plot(t,ym,'-k.',t,yr,'r-',t,0.25*(abs(ym-yr)),'--b','linewidth',3); 
legend('m-SNR','s-SNR',['r=',num2str(corr(ym',yr'),2),' ;e=',num2str(rmse(ym,yr),2)],'location','best');
% k-MEANS TO CLUSTER BASED ON COLOR
%     clc
%     face(:,:,1)=MouthZone.*double(im1(:,:,1));   
%     face(:,:,2)=MouthZone.*double(im1(:,:,2)); 
%     face(:,:,3)=MouthZone.*double(im1(:,:,3)); 
%     im=uint8(face);
% %     im=im1(VerticalAxis,HorizontalAxis,:);
%     lab_he = applycform(im, makecform('srgb2lab'));
%     ab1 = double(lab_he(:,:,2));
%     nrows = size(ab1,1); ncols = size(ab1,2);
%     ab = reshape(ab1,nrows*ncols,1);
%     nColors = 2;% repeat the clustering 3 times to avoid local minima
%     cluster_center=[ab1(round(yON+2*(yOFF-yON)/3),round(xOFF+xON)/2)
%                     ab1(1,1,1)];
%     [cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','start',cluster_center);%,'Replicates',7);
%     pixel_labels = reshape(cluster_idx,nrows,ncols);
%     segmented_images = cell(1,3);
%     rgb_label = repmat(pixel_labels,[1 1 3]);
%     for j = 1:nColors, color = im; color(rgb_label ~= j) = 0; segmented_images{j} = color; end
%     ff0=figure(11); imshow(segmented_images{1}), title('objects in cluster 1'); %set(ff0,'position',[500 750 400 200]);
%   % Filter and mask the image to find the area corresponding to lips:
%     LipColour = 195;%195
%     lab_he = applycform(im1, makecform('srgb2lab'));co=MouthZone.*double(lab_he(:,:,2)); LipColour=147;%cluster_center(1); 
%     hoho(k)=LipColour;

%% kalman filter ?% 
clear s; clc;
s.A = 1;
% % Define a process noise (stdev) of 2 volts as the car operates:
s.Q = 2^2; % variance, hence stdev^2
% Define the voltimeter to measure the voltage itself:
s.H = 1;
% Define a measurement error (stdev) of 2 volts:
s.R = 2^2; % variance, hence stdev^2
% Do not define any system input (control) functions:
s.B = 0;
s.u = 0;
% Do not specify an initial state:
s.x = nan;
s.P = nan;
% Generate random voltages and watch the filter operate.
tru=[]; % truth voltage
t=linspace(0,length(x)/fs,l_f);
l_t=length(t);
ma=normaNr(ma,min(Rms),max(Rms));
for n=1:l_t
   tru(end+1) = Rms(n);%
%    s(end).u = Rms(n);
%    s(end).R = 1000*(Rms(n)-ma(n))^2;
   s(end).z   = ma(n); % create a measurement
   s(end+1)=kalmanf(s(end)); % perform a Kalman filter iteration
end
yk=[s(2:end).x]; yn=[s(1:end-1).z]; ys=smooth(ma); yr=normaNr(Rms,min(ma),max(ma));
% plot measurement data:
figure(31); plot(t,tru,'k-',t,yn,'-r.',t,yk,'b-',t,ys,'y--'); hold on; %,t,yr,'g'
legend('RMS',['ma-SNR rmse=', num2str(rmse(tru,[s(1:end-1).z]),2)],['Kalman rmse=',num2str(rmse(tru,[s(2:end).x]),2)],'location','best');
hold off;
% clear s
% s.x = 12;
% s.A = 1;
% % % Define a process noise (stdev) of 2 volts as the car operates:
% s.Q = 2^2; % variance, hence stdev^2
% % Define the voltimeter to measure the voltage itself:
% s.H = 1;
% % Define a measurement error (stdev) of 2 volts:
% s.R = 2^2; % variance, hence stdev^2
% % Do not define any system input (control) functions:
% s.B = 0;
% s.u = 0;
% % Do not specify an initial state:
% s.x = nan;
% s.P = nan;
% % Generate random voltages and watch the filter operate.
% tru=[]; % truth voltage
% for t=1:20
%    tru(end+1) = randn*2+12;
%    s(end).z = tru(end) + randn*2; % create a measurement
%    s(end+1)=kalmanf(s(end)); % perform a Kalman filter iteration
% end
% % plot measurement data:
% figure(3); plot(1:20,tru,'k-',1:20,[s(1:end-1).z],'-r.',1:20,[s(2:end).x],'b-'); hold on;
% legend('True voltage',['Observ. rmse=', num2str(rmse(tru,[s(1:end-1).z]),2)],['Kalman rmse=',num2str(rmse(tru,[s(2:end).x]),2)]);
% title('Automobile Voltimeter Example'); hold off;
%% Compare the true and filtered responses graphically.
figure(4)
plot(t,ma,'--',t,y,'-'),
xlabel('time'), ylabel('Output')