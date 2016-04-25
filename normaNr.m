function [Y]=normaNr(x,m,M)
if size(x,2)==1
    Y=(x-min(x))/(max(x)-min(x))*(M-m)+m;
else
    X_m=min(min(x));
    X_M=max(max(x));
    Y=(x-X_m)/(X_M-X_m)*(M-m)+m;
end
end %EoF