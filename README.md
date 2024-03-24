> The release of EfficientSAM improved the speed of SAM, but there was no effective API to run everthing, point, and box modes.

According to the GitHub example of [EfficientSam](https://github.com/yformer/EfficientSAM) official, it is further packaged. The everthing pattern is further extended according to SAM.

# Install

1. install SAM

   ```shell
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

   or clone the repository locally and install with

   ```shell
   git clone git@github.com:facebookresearch/segment-anything.git
   cd segment-anything; pip install -e .
   ```

2. install EfficientSAM

   ```shell
   pip install git+https://github.com/yformer/EfficientSAM.git
   ```

   or clone the repository locally and install with

   ```shell
   git clone git+https://github.com/yformer/EfficientSAM.git
   cd EfficientSAM; pip install -e .
   ```

# Example

## EfficientSam Box Mode

```python
from eitsam_process.efficientsam_api import (
    get_efficient_sam_model,
    EfficientSAMPrompt,
    EfficientSAMEverthing
) 

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = get_efficient_sam_model(gpu=DEVICE)

box_generate = EfficientSAMPrompt(gpu,model)
input_point = np.array([[500,200], [750, 550]]) #[[x1, y1],[x2, y2]]

# input_label is used to indicate whether it is a point or a bounding box
# [[0,0]] is a positive point, [[1.1]] is a negative point, [[2,3]] is box
input_label = np.array([[2,3]])
masks = box_generate.segment_prompt(input_point, input_label, image_path="img.jpg")  

```

## EfficientSam Everthing Mode

```python
import cv2
import numpy as np

everthing_generate = EfficientSAMEverthing(grid_size=16,gpu=DEVICE, model=model)
masks = everthing_generate.segment_everthing(image_path="img.jpg")
print(len(masks))
for i,mask in enumerate(masks):
    if mask["area"] < 1000: continue
    cv2.imwrite(f"imgs/sub_img{i}.png", np.uint8(mask["segmentation"]*255))
    
```

The larger the `grid_size` setting, the more  memory is utilized. 

In the preliminary test, when `grid_size=16`, it will occupy about 15GB of V100 memory.

Configure `grid_size`  based on your memory conditions





## Citing

```latex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}


@article{xiong2023efficientsam,
  title={EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything},
  author={Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, Raghuraman Krishnamoorthi, Vikas Chandra},
  journal={arXiv:2312.00863},
  year={2023}
}
```

