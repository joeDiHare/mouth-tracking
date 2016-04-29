function ellipsePixels = draw_elips_im(imageSizeX,imageSizeY,centerX,centerY,radiusX,radiusY,Orient)
 %%
% imageSizeX = 640; imageSizeY = 480;
[columnsInImage rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
% Next create the ellipse in the image.
% centerX = 320; centerY = 240; radiusX = 250; radiusY = 150;
ellipsePixels = ...
    (rowsInImage - centerY).^2 ./ radiusY^2 + (columnsInImage - centerX).^2 ./ radiusX^2 <= 1;

ellipsePixels = imrotate(ellipsePixels,Orient,'crop');
% circlePixels is a 2D "logical" array.
% Now, display it with "image(ellipsePixels);"


%  a = 1/2*sqrt((x2-x1)^2+(y2-y1)^2);
%  b = a*sqrt(1-e^2);
%  t = linspace(0,2*pi);
%  X = a*cos(t);
%  Y = b*sin(t);
%  w = atan2(y2-y1,x2-x1);
%  x = (x1+x2)/2 + X*cos(w) - Y*sin(w);
%  y = (y1+y2)/2 + X*sin(w) + Y*cos(w);
%  plot(x,y,'y-')
%  axis equal