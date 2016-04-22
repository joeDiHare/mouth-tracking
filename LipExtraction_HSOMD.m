%% Extraction of lip information from a video
% Copyright(C) Joachim Gross and Hyojin Park
% Institue of Neuroscience and Psychology, University of Glasgow
% 28 April, 2015
%
% Modified by Heidi Solberg Økland and Matt Davis, October 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all;

%% Read video

VR      = VideoReader('sample.m4v');
video   = read(VR); % (VR, [x y]) = read in frame x to y
sz      = size(video); % resolution of video and number of frames
nFrames = sz(4); % number of frames in video

%% Adapt the script to your particular video

% 1. Crop the video so as to "zoom in" on the mouth:

% Find a window size that includes the mouth 
VerticalAxis = 380:670; HorizontalAxis = 520:810;
FrameNumber = 50;  % chosen to be a difficult frame
TestFrame = video(VerticalAxis,HorizontalAxis,:,FrameNumber);
figure; imshow(TestFrame); % look at it to check whether the frame size is good

% Create empty frames of a size that matches the TestFrame you decided on
WindowHeight = size(TestFrame,1);
WindowWidth = size(TestFrame,2);
Mouth = zeros(WindowHeight,WindowWidth,nFrames); 

% 2.  Set parameter values that make the lip colour stand out:

% Look at the red, green and blue pixels separately..
im1 = video(VerticalAxis,HorizontalAxis,:,FrameNumber); 
im1d = double(im1);
figure; subplot(2,2,1); imshow(im1(:,:,:));
subplot(2,2,2); imagesc(im1d(:,:,1)); colorbar; % 1 = red
subplot(2,2,3); imagesc(im1d(:,:,2)); colorbar; % 2 = green
subplot(2,2,4); imagesc(im1d(:,:,3)); colorbar; % 3 = blue
% .. and use the data cursor to find the values that best distinguishes lips from
% the rest, using approximately the same x/y coordinates in the three plots:
r_threshold = 170; g_threshold = 70; b_threshold = 70;

% Weight the image according to your choices..
im2 = (im1d(:,:,1)*r_threshold + im1d(:,:,2)*g_threshold + im1d(:,:,3)*b_threshold); % weight
im3 = sqrt(im1d(:,:,1).^2 + im1d(:,:,2).^2 + im1d(:,:,3).^2);
co = im2./im3;
figure; imagesc(co); colorbar; 
% ..and  use the data cursor to find the colour index that works best to single out the lips:
LipColour = 195;

% 3. Filter and mask the image to find the area of the frame corresponding
% to the lips:

fi = find(co > LipColour);
[a,b]   = sort(co(fi));
mask    = zeros(WindowHeight,WindowWidth);
mask(:) = 0;
mask(fi(b)) = 1;
h       = fspecial('average',[5 5]); % create averaging filter
se      = strel('disk',10);
tmp = imfilter(mask,h);
tmp(tmp < 0.5) = 0;
tmp(tmp > 0.5) = 1;
tmp = imclose(tmp,se);
B = bwconvhull(tmp,'object');
Mouth(:,:,FrameNumber) = B; 
figure; imagesc(B) % look at TestFrame and check that there is a lip blob 

%% Loop over frames to extract information about lip aperture

% Let's try and see if these parameters works for the whole video!
% Run this loop first, and then scroll down to write a video that (if all 
% has gone well) will show the "lip blob" on top of the original video frames.

LipStat   = zeros(4,nFrames); % create output variable
Orig   = zeros(WindowHeight,WindowWidth,3,nFrames,'uint8');

for k = 1:nFrames,
    
    im1 = video(VerticalAxis,HorizontalAxis,:,k); 
    im1d = double(im1);
    Orig(:,:,:,k) = im1;
    im2 = (im1d(:,:,1)*r_threshold + im1d(:,:,2)*g_threshold + im1d(:,:,3)*b_threshold);
    im3 = sqrt(im1d(:,:,1).^2 + im1d(:,:,2).^2 + im1d(:,:,3).^2);
    co = im2./im3;
    % figure; imagesc(co)
    fi = find(co > LipColour);
       
    [a,b]   = sort(co(fi));
    mask    = zeros(WindowHeight,WindowWidth);
    mask(:) = 0;
    mask(fi(b)) = 1;
    h       = fspecial('average',[5 5]); % create averaging filter
    se      = strel('disk',10);
    tmp = imfilter(mask,h);
    tmp(tmp < 0.5) = 0;
    tmp(tmp > 0.5) = 1;
    tmp = imclose(tmp,se);
    B = bwconvhull(tmp,'object');
    Mouth(:,:,k) = B; 
    % figure; imagesc(B)
    
    % Make a struct with information about Area, MajorAxis, MinorAxis for each blob
    st = regionprops(B,{'Area','MajorAxisLength','MinorAxisLength'});
    size_st = length(st);
    
    % Use e.g. MajorAxisLength to find the biggest blob (= the lips). Using
    % this variable works well because the mouth will usually be the widest
    % blob (whereas a neck shadow would often be a "tall", but not wide,
    % blob if you cropped the frame appropriately.)
    comp = zeros(1,length(st)); % make sure 'comp' is empty before we start.. (this was the bug in the original script)
    for x = 1:size_st;
        comp(x) = st(x).MajorAxisLength; 
    end
    maxval = max(comp);
    maxind = find(comp == maxval);
    
    % Extract information about the largest blob (using maxind)
    LipStat(1,k) = st(maxind).Area; % mouth area
    LipStat(2,k) = st(maxind).MajorAxisLength; % width of mouth opening
    LipStat(3,k) = st(maxind).MinorAxisLength; % height of mouth opening
    LipStat(4,k) = maxind; % tells you which field the info was extracted from - should be the same across all frames!
   
end

% Save the information
save LipStat LipStat

%% Plot things

samples = 1:nFrames;
Fs = 25;             
time = samples/Fs; % in seconds

MouthArea = LipStat(1,:);

plot(time,MouthArea)

%% Check each frame of the video

VO = VideoWriter('LipExtractionVideo','MPEG-4');
% VO.FrameRate = 50; % (default = 30)
open(VO);

for k = 1:nFrames,
    imshow(squeeze(Orig(:,:,:,k))); hold;
    title(['frame: ',num2str(k)]); % display frame number
    BD = bwboundaries(squeeze(Mouth(:,:,k)),'noholes'); 
    maxind = LipStat(4,k);
    plot(BD{maxind}(:,2),BD{maxind}(:,1),'k','LineWidth',2) % draw a line that represents the blob that was found in the loop above
    hold; drawnow;
    cF = getframe;
    writeVideo(VO,cF);
    pause % press a key to check every frame
end;
close(VO);



