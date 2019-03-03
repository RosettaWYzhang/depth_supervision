
function [depth_norm, confidence] = align_iasonas(depthim, rgbim, point2d_incolor)

sz= size(rgbim); height = sz(1); width = sz(2);

pts = point2d_incolor;
h = round(pts(:,1));
v = round(pts(:,2));
inside = (h>0)&(h<=width)&(v>0)&(v<height);
wt = find(inside&(~isnan(pts(:,1)))&(~isnan(pts(:,2))));
vals = depthim(wt);
h    = h(wt);
v    = v(wt);
%figure,scatter(pts(:,1),pts(:,2))
store = zeros(sz(1),sz(2));
for k=1:length(wt)
    store(v(k),h(k)) = vals(k);
end

positive = double(store>0);

scale = 3;
s = 4*scale;
[xs,ys] = meshgrid([-s:s],[-s:s]);
gauss_filt  = exp(-(xs.^2 + ys.^2)/(2*scale^2));
gauss_filt = gauss_filt/sum(gauss_filt(:));


support_smooth = conv2(positive,gauss_filt,'same');
value_smooth   = conv2(store,gauss_filt,'same');
depth_estimate = value_smooth./(max(support_smooth,.01));

%local_variance  = conv2((store - positive.*depth_estimate).^2,gauss_filt,'same')./(max(support_smooth,cutoff));
%cutoff = 3.;
%depth_estimate = value_smooth./(max(support_smooth,cutoff));
%confidence = support_smooth>cutoff;
confidence = (support_smooth<.13) & (support_smooth>.07);

mask = zeros(height,width);
mask(200:900,700:1200) = 1;

center_depth = depth_estimate(500,900);
dist_depth = abs(depth_estimate  - center_depth);
dist_depth = dist_depth  + 300*(1-mask);
wt  = find(dist_depth<300);
[i,j]=  find(dist_depth<300);
val = depth_estimate(wt);

szs = 30;
chk =  double(checkerboard(szs,1080/(2*szs),1920/(2*szs)) >.5);


depth_norm = depth_estimate/max(depth_estimate(:));

composite = chk.*depth_norm + (1.-chk).*double(confidence);
%composite = chk.*depth_norm + (1.-chk).*double(rgb2gray(rgbim))/256;
% composite = chk.*confidence + (1.-chk).*double(rgb2gray(rgbim))/256;
end

