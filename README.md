# augmentation_example
augmentation_example

## 아나콘다 환경 복사 
1. export  (online)
2. 복사     (offline)

```
conda env export -n test > environment.yaml
```

```
conda env create -f environment.yaml
```
```
transforms = A.Compose([
    A.OneOf([
        A.RandomSizedBBoxSafeCrop(height = 480,width = 640,erosion_rate =  0.0, interpolation = 1,p=0.7),
        A.RandomSizedBBoxSafeCrop(height = 960,width = 1280,erosion_rate =  0.0, interpolation = 1,p=0.7),
        A.RandomSizedBBoxSafeCrop(height = 1440,width = 1920,erosion_rate =  0.0, interpolation = 1,p=0.7),
        A.RandomSizedBBoxSafeCrop(height = 1920,width = 2560,erosion_rate =  0.0, interpolation = 1,p=0.7),
        A.RandomSizedBBoxSafeCrop(height = 2400,width = 3200,erosion_rate =  0.0, interpolation = 1,p=0.7),
    ]),
    A.Rotate (limit=15, interpolation=1, border_mode=1, value=None, mask_value=None, always_apply=False, p=0.2),
    A.HueSaturationValue(hue_shift_limit=(-5,5), sat_shift_limit=(0,15), val_shift_limit=(0,15), always_apply=False, p=0.7), 
    A.Blur(blur_limit=2, always_apply=False, p=0.5),
    A.CLAHE (clip_limit=1, tile_grid_size=(8, 8), always_apply=False, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=[-0.1,0.01], contrast_limit=[0,0.3],p=0.9),
    A.RandomShadow (shadow_roi=(0.3, 0.3, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, always_apply=False, p=0.6),
    A.RandomSunFlare (flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=0, num_flare_circles_upper=2, src_radius=50, src_color=(255, 255, 255), always_apply=False, p=0.5)

    ],
    bbox_params=A.BboxParams(format="yolo"))
```

pip3 install numpy  
pip3 install scikit-build  
pip3 install -U albumentation  

![image](https://user-images.githubusercontent.com/52307552/147082502-d5f1abc1-610d-4029-a8c1-d0ef38b4fa7c.png)
