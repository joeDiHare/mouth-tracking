function P = bwconvhull_alt(BW)

% usage: P is a binary image wherein the convex hull of objects are returned, BW is the input binary image
% P= bwconvhull_alt(BW);

warning off all
s=regionprops(logical(BW),'ConvexImage','BoundingBox');
P=zeros(size(BW));
for no=1:length(s)
P(s(no).BoundingBox(2):s(no).BoundingBox(2)+s(no).BoundingBox(4)-1,...
    s(no).BoundingBox:s(no).BoundingBox(1)+s(no).BoundingBox(3)-1)=s(no).ConvexImage;
end
warning on all