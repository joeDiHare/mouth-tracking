%%% CamKas [SC] March-16
%%
clear all; clc; close all;
load('svmStruct2.mat');
DBFILEPATH='C:\Users\sc04\Documents\MATLAB\VidTIMIT_mouthVals\';
files = dir(DBFILEPATH); files = files(3:end);
l_fS=length(files); %no. of files
SNR_vals=[-20:5:20 Inf]; l_SNR=length(SNR_vals);
k=0;
for tt=1:2%l_fS
    file  = files(tt).name;
    subj  = file(5:9);
    sent  = file(11:end-4);
    load(strcat(DBFILEPATH,'MTk_',subj,'_',sent,'.mat'), 'output');
             
    speech.clean=output.x;
    time =output.t;
    l_t  =length(time);
    fs   =output.fs;
    MTR  =normaNr(output.LipStat(2,:),0,1);
    for snr=SNR_vals(3:end)
        k=k+1;
        fprintf(1,'\n%.1f%% (%d of %d)\tTalker: %s \tSent: %s\tSNR: %d',(100*k/(l_SNR*l_fS)),k,l_SNR*l_fS,subj,sent,snr);        
        speech.noise = SNRadjustNOISE(speech.clean,randn(size(speech.clean)),snr);
        speech.noisy  = speech.clean + speech.noise;
        
        Tw = time(end)/length(time)*1e3;    % analysis frame duration (ms) 
        Ts = Tw/8;  % analysis frame shift (ms)
        LC = -20;    % local SNR criterion (LC) 
         % apply ideal binary mask to the noisy speech signal 
        speech.procesUNP = idbm( speech.noisy, speech.clean, fs, Tw, Ts, -Inf ); 
        [speech.procesIBM, MASK] = idbm( speech.noisy, speech.clean, fs, Tw, Ts, LC ); 
        IBM=normaNr(smooth(mean(MASK(:,1e2:1e3),2)),0,1);
%         MTR=resample(MTR,length(IBM),length(MTR))';
        Rms=zeros(l_t,1); N=round(Tw*1e-3*fs); for n=1:l_t, Rms(n) = rms(speech.noisy(1+(n-1)*N:n*N)); end
%         FVthis=[time' output.LipStat([1],:)' Rms];    
%         MTR = svmclassify(svmStruct,FVthis)';
% %         s1=floor((length( speech.clean )- round( fs*Tw*0.001 ))/round( fs*Ts*0.001 )+1);
%         MTRt=round(resample(MTR,size(MASK,1),length(MTR)))'*ones(1,size(MASK,2));
% %         MTRt=resample(MTR,length(speech.clean),length(MTR));               
%         MTRt(:,2000:end)=1;
%         speech.procesMTR = MASKtoY( speech.noisy, MTRt, fs, Tw, Ts );%( (.5+.5*MTRt).*speech.noisy' );       
        IBM=normaNr(smooth(nanmean(MASK(:,1:1e3),2)),0,1);
        Th=1; while (sum(IBM>Th)/numel(IBM)<=.60 && Th>0), Th=Th-0.01; end
        IBMb=zeros(size(MASK)); IBMb(IBM>=Th,:)=1;
        speech.procesMTR = MASKtoY( speech.noisy, IBMb, fs, Tw, Ts );%( (.5+.5*MTRt).*speech.noisy' );       
        
        ncm_val1=NCMestimation(speech.clean, speech.procesUNP,  fs);
        ncm_val2=NCMestimation(speech.clean, speech.procesIBM, fs);
        ncm_val3=NCMestimation(speech.clean, speech.procesMTR,  fs);
        
        R(k,:)=[k snr ncm_val1 ncm_val2 ncm_val3];
        fprintf(1,'\tIBM: %.1f; \tMTK: %.1f; \tUNP: %.1f;',ncm_val2,ncm_val3,ncm_val1);        

    end            
end
usageProc='speech.procesMTR = MASKtoY( speech.noisy, MTRt, fs, Tw, Ts )';
% res=R; save('MTK_res4.mat'); 
%% plot RES
clear all; close all; clc; load('MTK_res4.mat'); 
R=res; R(R(:,2)==Inf,2)=50; SNR_vals(end)=50;
for k=1:length(SNR_vals)
    mR(k,:)=[SNR_vals(k) nanmean(R(R(:,2)==SNR_vals(k), 3:5))];
end
ord = @(n) repmat(SNR_vals,1,size(n,1)/length(SNR_vals))+0.4*randn(1,size(n,1));
figure(1);
% for l=1:9 %ord( R(100*(l-1)+1:l*100,1)), R(100*(l-1)+1:l*100,3),'.k',...
plot(ord(mR),mR(:,2),'-k',...
     ord(mR),mR(:,3),'-b',...
     ord(mR),mR(:,4),'-r','linewidth',5); hold on;
plot(ord( R), R(:,3),'.k',...
     ord( R), R(:,4),'.b',...
     ord( R), R(:,5),'.r'); hold off;
xlim([-25 54]);
% pause; end;
legend('UNP','IBM','MTK','location','best');
A=cell(2,1); A{1} = SNR_vals(1:end-1); A{2} = '+Inf'; set(gca,'xtick',SNR_vals,'xticklabel',A);
xlabel('snr [dB]'); ylabel('percentage score'); 

%% label videos
clear all; clc; close all;
DBFILEPATH='U:\docs\VidTIMIT\';
talkers = dir(DBFILEPATH); talkers = talkers(3:end);
l_TALKERS=length(talkers); %no. of talkers
for tt=10:l_TALKERS
    subj  = talkers(tt).name;
    sents = dir(strcat(DBFILEPATH, subj,'\video')); sents = sents(3:end);
    l_SENTS = length(sents);
    for ss=1:l_SENTS
        sent=sents(ss).name;
        if ~any([strcmp(sent,'head') strcmp(sent,'head2') strcmp(sent,'head3')])
        filepath=strcat(DBFILEPATH,subj,'\video\',sent);
        fprintf(1,'\nTalker: %s (%d of %d)\tSent: %s (%d of %d)',subj,tt,l_TALKERS,sent,ss,l_SENTS);
        output = lipTracker(filepath,ss==4);
%         save(strcat('MTk_',subj,'_',sent), 'output');
        end
    end
end

%% TRAIN: SVM binary classification
clear all; clc; close all;
DBFILEPATH='C:\Users\sc04\Documents\MATLAB\VidTIMIT_mouthVals\';
files = dir(DBFILEPATH); files = files(3:end);
l_fS=length(files); %no. of files
SNR_vals=[-20:5:20 Inf]; l_SNR=length(SNR_vals);
k=0; LB=[]; FV=[]; tic
for tt=1:l_fS/2
    file  = files(tt).name;
    subj  = file(5:9);
    sent  = file(11:end-4);
    load(strcat(DBFILEPATH,'MTk_',subj,'_',sent,'.mat'), 'output');
             
    speech.clean=output.x;
    time =output.t;
    l_t  =length(time);
    fs   =output.fs;
    MTR  =normaNr(output.LipStat(2,:),0,1);
    for snr=SNR_vals
        k=k+1;
        fprintf(1,'\n%.1f%% (%d of %d)\tTalker: %s \tSent: %s\tSNR: %d',(100*k/(l_SNR*l_fS)),k,l_SNR*l_fS,subj,sent,snr);        
        speech.noise = SNRadjustNOISE(speech.clean,randn(size(speech.clean)),snr);
        speech.noisy  = speech.clean + speech.noise;
        
        Tw = time(end)/length(time)*1e3;    % analysis frame duration (ms) 
        Ts = Tw;  % analysis frame shift (ms)
        LC = -20;    % local SNR criterion (LC) 
         % apply ideal binary mask to the noisy speech signal 
        speech.procesUNP = idbm( speech.noisy, speech.clean, fs, Tw, Ts, -Inf ); 
        [speech.processIBM, MASK] = idbm( speech.noisy, speech.clean, fs, Tw, Ts, LC ); 
%         IBM=resample(normaNr(round(1.5*smooth(nanmean(MASK(:,[150:300 800:1000]),2))),0,1),size(MASK,1),length(MTR))';
          IBM=resample((normaNr(smooth(nanmean(MASK(:,1:1e3),2)),0,1)),size(MASK,1),length(MTR))';
          Th=1; while (sum(IBM>Th)/numel(IBM)<=.60 && Th>0), Th=Th-0.01; end
          IBMb=zeros(size(IBM)); IBMb(IBM>=Th)=1;
%         MTR=resample(MTR,length(IBM),length(MTR))';
%         MTRt=resample(MTR,length(speech.clean),length(MTR));
        Rms=zeros(size(IBM(:))); N=round(Tw*1e-3*fs); for n=1:l_t, Rms(n) = rms(speech.noisy(1+(n-1)*N:n*N)); end
        
        FV=[FV; time' output.LipStat([1],:)' Rms];
        LB=[LB; IBMb(:)];
%         plot(1:l_t,normaNr(output.LipStat(1,:),0,1),'k',1:l_t,normaNr(Rms,0,1),1:l_t,IBMb,'o'); ylim([-.1 1.1]);
%         pause
    end            
end
% train
xdata = FV(~isnan(LB),:);
group = LB(~isnan(LB));
svmStruct = svmtrain(xdata(1:end,:),group(1:end,:));
toc
% Xnew = xdata;
% species = svmclassify(svmStruct,Xnew,'ShowPlot',true)

%% test
tt=l_fS/2+1
file  = files(tt).name;
subj  = file(5:9);
sent  = file(11:end-4);
load(strcat(DBFILEPATH,'MTk_',subj,'_',sent,'.mat'), 'output');

speech.clean=output.x;
time =output.t;
fs   =output.fs;
MTR  =normaNr(output.LipStat(2,:),0,1);
for snr=SNR_vals
    k=k+1;
    fprintf(1,'\n%.1f%% (%d of %d)\tTalker: %s \tSent: %s\tSNR: %d',(100*k/(l_SNR*l_fS)),k,l_SNR*l_fS,subj,sent,snr);        
    speech.noise = SNRadjustNOISE(speech.clean,randn(size(speech.clean)),snr);
    speech.noisy  = speech.clean + speech.noise;

    Tw = time(end)/length(time)*1e3;    % analysis frame duration (ms) 
    Ts = Tw;  % analysis frame shift (ms)
    LC = -20;    % local SNR criterion (LC) 
     % apply ideal binary mask to the noisy speech signal 
    speech.procesUNP = idbm( speech.noisy, speech.clean, fs, Tw, Ts, -Inf ); 
    [speech.processIBM, MASK] = idbm( speech.noisy, speech.clean, fs, Tw, Ts, LC ); 
%         IBM=resample(normaNr(round(1.5*smooth(nanmean(MASK(:,[150:300 800:1000]),2))),0,1),size(MASK,1),length(MTR))';
      IBM=resample(round(normaNr(10*smooth(nanmean(MASK(:,1:end/2),2)),0,1)),size(MASK,1),length(MTR))';
%         MTR=resample(MTR,length(IBM),length(MTR))';
%         MTRt=resample(MTR,length(speech.clean),length(MTR));
    Rms=zeros(size(IBM(:))); N=round(Tw*1e-3*fs); for n=1:length(time), Rms(n) = rms(speech.noisy(1+(n-1)*N:n*N)); end

    Xnew = [time' output.LipStat([1],:)' Rms];     
    species = svmclassify(svmStruct,Xnew,'ShowPlot',true)'

end            
% plot(Xnew(:,1),Xnew(:,2),'ro','MarkerSize',12);


%% process video

% %save(strcat('R-',subj,'-',sent),'R');
% ym=normaNr(ma,min(Rms),max(Rms)); yr=smooth(Rms)';
% figure(31); plot(t,ym,'-k.',t,yr,'r-',t,0.25*(abs(ym-yr)),'--b','linewidth',3); 
% legend('m-SNR','s-SNR',['r=',num2str(corr(ym',yr'),2),' ;e=',num2str(rmse(ym,yr),2)],'location','best');

% subj='fadg0'; sent='sa1';
% subj='faks0'; sent='sx403';
% subj='faks0'; sent='sa1';
%%
% pathV=strcat('C:\Users\sc04\Documents\MATLAB\',subj,'\video\',sent);
% pathA=strcat('C:\Users\sc04\Documents\MATLAB\',subj,'\audio\',sent,'.wav');
% %% compute mouth area using lip-tracking
% % first select area
% clc; close(figure(1));
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
% 
% %%% estimate lipcolor
% r_threshold = 170; g_threshold = 70; b_threshold = 70;    
% im2 = MouthZone.*(im1d(:,:,1)*r_threshold + im1d(:,:,2)*g_threshold + im1d(:,:,3)*b_threshold); % weight
% im3 = MouthZone.*sqrt(im1d(:,:,1).^2 + im1d(:,:,2).^2 + im1d(:,:,3).^2);
% co = im2./(im3+eps);
% [a,b]=hist(co(co>0),100);
% [u,sig,tdfj,iter] = fit_mix_gaussian_multiple(b',2);
% LipColour = max(u);
% %%
% % clear all; clc; close all; load('lip_data');
% f1=figure(1); set(f1,'position',[150 100 900 500]);  
% LipStat=NaN(7,length(items));
% for k=1:length(items)
%     im1 = imread(strcat(pathV,'\',items(k).name));   
%     im1d= double(im1(:,:,:));
% 
%     r_threshold = 170; g_threshold = 70; b_threshold = 70;    
%     im2 = MouthZone.*(im1d(:,:,1)*r_threshold + im1d(:,:,2)*g_threshold + im1d(:,:,3)*b_threshold); % weight
%     im3 = MouthZone.*sqrt(im1d(:,:,1).^2 + im1d(:,:,2).^2 + im1d(:,:,3).^2);
%     co = im2./(im3+eps);
%     
%     fi = find(co > LipColour);
%     [a,b]   = sort(co(fi));
%     mask    = zeros(size(co));
%     mask(fi(b)) = 1;
%     h       = fspecial('average',[5 5]); % create averaging filter
%     se      = strel('disk',10);
%     tmp = imfilter(mask,h);
%     tmp(tmp < 0.5)  = 0;
%     tmp(tmp >= 0.5) = 1;
%     tmp = imclose(tmp,se);
%     B = bwconvhull_alt(tmp);
%     %100*sum(B(:)==1)/numel(B), imagesc(B);
%     
%     st = regionprops(B,{'Area','MajorAxisLength','MinorAxisLength','Centroid'});
%     size_st = length(st);
%     
%     % Use e.g. MajorAxisLength to find the biggest blob (= the lips). 
%     comp = zeros(1,length(st)); 
%     for x = 1:size_st, comp(x) = st(x).MajorAxisLength; end
%     maxval = max(comp);
%     maxind = find(comp == maxval);
%     
%     % Extract information about the largest blob (using maxind)
%     LipStat(1,k) = st(maxind).Area; % mouth area
%     LipStat(2,k) = st(maxind).MajorAxisLength; % width of mouth opening
%     LipStat(3,k) = st(maxind).MinorAxisLength; % height of mouth opening
%     LipStat(4,k) = maxind; % tells you which field the info was extracted from - should be the same across all frames!
%     LipStat(5,k) = mean2(imfilter(co.*B, fspecial('laplacian'), 'replicate', 'conv').^2); %extract color gradient
%     LipStat(6,k) =  st(maxind).Centroid(1);
%     LipStat(7,k) =  st(maxind).Centroid(2);
% 
% % Apply smoothness constraints
%     if pdist([centroids LipStat([6 7],k)]','euclidean')<10
%         MouthZone = imdilate(B,strel('disk',8));
%     end
%         
% %     display
%     face(:,:,1)=double(im1(:,:,1)).*(1-(imdilate(MouthZone,strel('disk',4))-MouthZone));   
%     face(:,:,2)=double(im1(:,:,2)).*(1-(imdilate(MouthZone,strel('disk',4))-MouthZone)); 
%     face(:,:,3)=double(im1(:,:,3)).*(1-(imdilate(MouthZone,strel('disk',4))-MouthZone)); 
%     subplot(121); imshow(uint8(face)); title('search area')
%     face(:,:,1)=double(im1(:,:,1)).*(1-(imdilate(B,strel('disk',4))-B));   
%     face(:,:,2)=double(im1(:,:,2)).*(1-(imdilate(B,strel('disk',4))-B)); 
%     face(:,:,3)=double(im1(:,:,3)).*(1-(imdilate(B,strel('disk',4))-B)); 
%     subplot(122); imshow(uint8(face)); title('seleced mouth area');%     ff1=figure(12); imshow(face_a); set(ff1,'position',[150 100 500 500]);
%     pause(1e-4);
%   
% end
% ma=LipStat(1,:);
%%
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

% %% kalman filter ?% 
% clear s; clc;
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
% t=linspace(0,length(x)/fs,l_f);
% l_t=length(t);
% ma=normaNr(ma,min(Rms),max(Rms));
% for n=1:l_t
%    tru(end+1) = Rms(n);%
% %    s(end).u = Rms(n);
% %    s(end).R = 1000*(Rms(n)-ma(n))^2;
%    s(end).z   = ma(n); % create a measurement
%    s(end+1)=kalmanf(s(end)); % perform a Kalman filter iteration
% end
% yk=[s(2:end).x]; yn=[s(1:end-1).z]; ys=smooth(ma); yr=normaNr(Rms,min(ma),max(ma));
% % plot measurement data:
% figure(31); plot(t,tru,'k-',t,yn,'-r.',t,yk,'b-',t,ys,'y--'); hold on; %,t,yr,'g'
% legend('RMS',['ma-SNR rmse=', num2str(rmse(tru,[s(1:end-1).z]),2)],['Kalman rmse=',num2str(rmse(tru,[s(2:end).x]),2)],'location','best');
% hold off;
% %% Compare the true and filtered responses graphically.
% figure(4)
% plot(t,ma,'--',t,y,'-'),
% xlabel('time'), ylabel('Output')