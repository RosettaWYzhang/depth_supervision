% this script generates ground truth and aligns it with rgb space
% depth, confidence and rgb images are saved in specified directories
% (ground truth for hd does not produce sensible results)
align_sensor = 'rgb';
cam_index = '50_01';
idk = 1; % kinect camera ID
start_frame = 500;
end_frame = 510;
hd_index_list= start_frame:end_frame; % Target frames you want to export ply files
depth_cell = cell([1, end_frame - start_frame + 1]);

%Relative Paths
root_path = '/home/wanyue/Desktop/panoptic-toolbox'; %Put your root path where sequence folders are locates
seqName = '161029_flute1';  %Put your target sequence name here
kinectImgDir = sprintf('%s/%s/kinectImgs',root_path,seqName); 
hdImgDir = sprintf('%s/%s/hdImgs',root_path,seqName);
kinectDepthDir = sprintf('%s/%s/kinect_shared_depth',root_path,seqName);
calibFileName = sprintf('%s/%s/kcalibration_%s.json',root_path,seqName,seqName);
syncTableFileName = sprintf('%s/%s/ksynctables_%s.json',root_path,seqName,seqName);
panopcalibFileName = sprintf('%s/%s/calibration_%s.json',root_path,seqName,seqName);
panopSyncTableFileName = sprintf('%s/%s/synctables_%s.json',root_path,seqName,seqName);

% Output folder Path
% Change the following if you want to save outputs on another folder
depthOutputDir=sprintf('%s/%s/kinoptic_depth_%s',root_path,seqName, align_sensor);
rgbOutputDir=sprintf('%s/%s/kinoptic_rgb_%s',root_path,seqName, align_sensor);
mkdir(depthOutputDir);
mkdir(rgbOutputDir);
fprintf('rgb images will be saved in: %s\n', rgbOutputDir);
fprintf('depth images will be saved in: %s\n',depthOutputDir);

%Other parameters
bVisOutput = 1; %Turn on, if you want to visualize what's going on
bRemoveFloor= 1;  %Turn on, if you want to remove points from floor
floorHeightThreshold = 0.5; % Adjust this (0.5cm ~ 7cm), if floor points are not succesfully removed
                            % Icreasing this may remove feet of people
bRemoveWalls = 1; %Turn on, if you want to remove points from dome surface
addpath('jsonlab');
addpath('kinoptic-tools');

 
%% Load syncTables
ksync = loadjson(syncTableFileName);
knames = {};
for id=1:10; knames{id} = sprintf('KINECTNODE%d', id); end
psync = loadjson(panopSyncTableFileName); %%Panoptic Sync Tables


%% Load Kinect Calibration File
kinect_calibration = loadjson(calibFileName); 
panoptic_calibration = loadjson(panopcalibFileName);
panoptic_camNames = cellfun( @(X) X.name, panoptic_calibration.cameras, 'uni', false ); %To search the targetCam


hd_index_list = hd_index_list+2; %This is the output frame (-2 is some weired offset in synctables)

for hd_index = hd_index_list
    hd_index_afterOffest = hd_index-2; %This is the output frame (-2 is some weired offset in synctables)
    out_fileName = sprintf('%s/depth%08d.jpg', depthOutputDir, hd_index_afterOffest);
    out_confidence_fileName = sprintf('%s/confidence%08d.jpg', depthOutputDir, hd_index_afterOffest);
    out_rgb_fileName = sprintf('%s/rgb%08d.jpg', rgbOutputDir, hd_index_afterOffest);
    %% Compute Universal time
    selUnivTime = psync.hd.univ_time(hd_index);
    fprintf('hd_index: %d, UnivTime: %.3f\n', hd_index, selUnivTime)

    %% Select corresponding frame index rgb and depth by selUnivTime
    % Note that kinects are not perfectly synchronized (it's not possible),
    % and we need to consider offset from the selcUnivTime
    [time_distc, cindex] = min( abs( selUnivTime - (ksync.kinect.color.(knames{idk}).univ_time-6.25) ) );  %cindex: 1 based
    ksync.kinect.color.(knames{idk}).univ_time(cindex);
    [time_distd, dindex] = min( abs( selUnivTime - ksync.kinect.depth.(knames{idk}).univ_time ) ); %dindex: 1 based
    
    % Filtering if current kinect data is far from the selected time
    fprintf('idk: %d, %.4f\n', idk, selUnivTime - ksync.kinect.depth.(knames{idk}).univ_time(dindex));
    if abs(ksync.kinect.depth.(knames{idk}).univ_time(dindex) - ksync.kinect.color.(knames{idk}).univ_time(cindex))>6.5
        fprintf('Skipping %d, depth-color diff %.3f\n',  abs(ksync.kinect.depth.(knames{idk}).univ_time(dindex) - ksync.kinect.color.(knames{idk}).univ_time(cindex)));    
        continue;
    end

    if time_distc>30 || time_distd>17 
        fprintf('Skipping %d\n', idk);
        [time_distc, time_distd];
        continue;
    end

    % Extract image and depth
    rgbFileName = sprintf('%s/%s/%s_%08d.jpg',kinectImgDir,cam_index, cam_index, cindex);
    rgbim = imread(rgbFileName); % cindex: 1 based
    camCalibDataKinect = kinect_calibration.sensors{idk};
    depthFileName = sprintf('%s/KINECTNODE%d/depthdata.dat',kinectDepthDir,idk);
    depthim = readDepthIndex_1basedIdx(depthFileName,dindex);  % cindex: 1 based  
    % Align depth image using the Iasonas' interpolation Method
    [~, point2d_incolor] = unprojectDepth_release(depthim, camCalibDataKinect, true);
    [depthim_aligned, confidence] = align_iasonas(depthim, rgbim, point2d_incolor);


    % save aligned ground truth depth, confidence and the corresponding synced rgb images
    imwrite(depthim_aligned, out_fileName);
    imwrite(confidence, out_confidence_fileName);
    imwrite(rgbim, out_rgb_fileName);
end



