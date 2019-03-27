### example configuration to run cmu_depth.py

- DATASET: 'cmu_depth'  
  IMAGE_ROOT:   '/home/wanyue/Desktop/panoptic-toolbox/171204_pose3'  
  ANNOTATION_FILE: '/home/wanyue/Desktop/panoptic-toolbox/python/cmu_simple_depth.json'  
  FLIP: true  
  ROT_FACTOR: 30  
  SCALE_FACTOR: 0.25  
  SAMPLING_RATIO: 1  

### add in init.py
```python
from .cmu_depth import CMUDepthDataset as cmu_depth
```
### example code
  ```python
  import dataset.cmu_depth as loader
  import yaml
  from core.config import config
  from core.config import update_config
  update_config("/home/wanyue/Desktop/ariel_seg/experiments/holopose_part/holopose_body_integral.yaml")
  data_loader = loader(config, config.TRAIN_DATASETS[0], True, transform=None, dotiny=False)
  ```
