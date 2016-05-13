# mouth-tracking

This script performs a computation of the mouth area (including lips)
using mostly hue. 

Check the demo here >> https://youtu.be/EcEZTZqkrK0 << 

The algorithm performs a computation of the colour
clusters (mostly skin and lip) using k-means algorithm. The standard
deviation of these clusters is estimated by fitting up to 5 gaussian
distributions to the histogram of the color leveles (optimization done
with E.M.). To improve the algorithm, some smoothness is applied
between frames, so that the area cannot vary >60% between consecutive
frames, and the "search area" for the lips also varies accordingly. 
The user needs to initiate the algorithm by selecting the search area in
the first frame using the prompted GUI.

This algorithm was tested ONLY with the VidTIMIT database. If your videos
have different format, you'll probably need to modify the code accordingly. 

input:   
- filepath. This refers to a location that containts both video and
  audio. For VidTIMIT, frames and audio are in separate folders. 
  (See example below.)
- flag_plot. Plot frame-by-frame the results of lip detection/traking.
- flag_manual. When the algorithm fails, run with flag_manual=true to
  check that each frame is labelled correctly and, if necessary, correct 
  the mouth area.

With flag_manual=true, the user will be prompted to check that the
correct mouth area has been selected. If the user intends to select a
different area, the user needs to click twice in the subplot 2.24. 
This will display a red square and the algorithm will re-run to detect
the mouth for that frame. 
REMEMBER to always press OK after after selecting the new mouth-area, or 
if you're already happy with estimated area.

Example: How to run this script.

```matlab
clear all; clc; close all;
DBFILEPATH='C:\MATLAB\VidTIMIT\';
talkers = dir(DBFILEPATH); talkers = talkers(3:end);
l_TALKERS=length(talkers); %no. of talkers

for tt=1:l_TALKERS
    subj  = talkers(tt).name;
    sents = dir(strcat(DBFILEPATH, subj,'\video')); sents = sents(3:end);
    l_SENTS = length(sents);
    for ss=1:l_SENTS%ss=4
        sent=sents(ss).name;
        if ~any([strcmp(sent,'head') strcmp(sent,'head2') strcmp(sent,'head3')]) 
        % exclude folders 'head', 'head2' and 'head3'.
        filepath=strcat(DBFILEPATH,subj,'\video\',sent);
        fprintf(1,'\nTalker: %s (%d of %d)\tSent: %s (%d of %d)',subj,tt,l_TALKERS,sent,ss,l_SENTS);
        output = lipTracker(filepath, true, true);
%         save(strcat('MTk_',subj,'_',sent), 'output');
        end
    end
end
```
