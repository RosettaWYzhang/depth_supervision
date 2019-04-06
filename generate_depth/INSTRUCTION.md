# Instruction for extracting ground truth depth from CMU Panoptic

1 . clone CMU panoptic toolbox

`git clone https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox.git`

2 . download all kinoptic sequences

`./scripts/getData_kinoptic.sh`

or just download one sequence

`./scripts/getData_kinoptic.sh 160422_haggling1`


3 . extract all rgb images from videos

`cd 160422_haggling1
../scripts/kinectImgsExtractor.sh`

4 . put generate_rgb_depth.m and align_iasonas_rgb.m in this repo into panoptic-toolbox/matlab/.

5 . Before running generate_rgb_depth.m, specify sequence name, start and end frame, root path and output directories.

6 . To generate the annotation file for Ariel Dataset CMUDepth, bounding boxes are needed from mask-rcnn.

`git clone https://github.com/facebookresearch/maskrcnn-benchmark.git`  
put bounding_box_for_depth_supervision.ipynb in maskrcnn-benchmark/demo folder. Update sequence name and output directory before running the file. This will save the bounding boxes for each image in a separate .npy file.

7 . put cmu_depth_parsing.ipynb in panoptic-toolbox/python/. Update the sequence name in the file.  The output will be an annotation file needed for training. 
