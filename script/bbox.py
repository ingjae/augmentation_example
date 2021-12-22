import numpy as np
import albumentations as A
import random
import cv2 
import matplotlib.pyplot as plt
import names
import dataset
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def draw_bboxes(img, bboxes):
    height, width = img.shape[:2]
    for x, y, w, h, label in bboxes:
        class_name = names.category_id_to_name[label]
        x *= width
        y *= height
        w *= width
        h *= height

        x1 = int(x - w / 2 + 1)
        x2 = int(x1 + w)
        y1 = int(y - h / 2 + 1)
        y2 = int(y1 + h)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=height // 100)
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)  
        # cv2.rectangle(img, (x1, y1 - int(1.3 * text_height)), (x1 + text_width, y1), BOX_COLOR, -1)
        
        cv2.putText(
            img,
            text=class_name,
            org=(x1, y1 - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.2, 
            color=TEXT_COLOR, 
            lineType=cv2.LINE_AA,
        )
def read_image():
    pass
def read_bboxes():
    pass
def visualize_bboxes(grid_index,grid_title,image,bboxes):
    plt.subplot(grid_index, title=grid_title)
    draw_bboxes(image, bboxes)
    plt.imshow(image, vmin=0, vmax=255)


image = plt.imread(dataset.panel)
bboxes = dataset.panel_label


# random.seed(43)
# np.random.seed(0)

transforms = A.Compose([
    A.OneOf([
        A.RandomSizedBBoxSafeCrop(height = 480,width = 640,erosion_rate =  0.0, interpolation = 1,p=0.7),
        A.RandomSizedBBoxSafeCrop(height = 960,width = 1280,erosion_rate =  0.0, interpolation = 1,p=0.7),
        A.RandomSizedBBoxSafeCrop(height = 1440,width = 1920,erosion_rate =  0.0, interpolation = 1,p=0.7),
        A.RandomSizedBBoxSafeCrop(height = 1920,width = 2560,erosion_rate =  0.0, interpolation = 1,p=0.7),
        A.RandomSizedBBoxSafeCrop(height = 2400,width = 3200,erosion_rate =  0.0, interpolation = 1,p=0.7),
    ]),
    A.Rotate (limit=15, interpolation=1, border_mode=1, value=None, mask_value=None, always_apply=False, p=0.2),
    A.HueSaturationValue(hue_shift_limit=(-5,5), sat_shift_limit=(-5,15), val_shift_limit=(0,15), always_apply=False, p=0.7),
    A.Blur(blur_limit=2, always_apply=False, p=0.5),
    A.CLAHE (clip_limit=1, tile_grid_size=(8, 8), always_apply=False, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=[-0.1,0.01], contrast_limit=[0,0.3],p=0.9),
    A.RandomShadow (shadow_roi=(0.3, 0.3, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, always_apply=False, p=0.6),
    A.RandomSunFlare (flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=0, num_flare_circles_upper=2, src_radius=50, src_color=(255, 255, 255), always_apply=False, p=0.5)

    ],
    bbox_params=A.BboxParams(format="yolo"))
    
res1 = transforms(image=image, bboxes=bboxes)
res2 = transforms(image=image, bboxes=bboxes)
res3 = transforms(image=image, bboxes=bboxes)
res4 = transforms(image=image, bboxes=bboxes)
res5 = transforms(image=image, bboxes=bboxes)
res6 = transforms(image=image, bboxes=bboxes)
res7 = transforms(image=image, bboxes=bboxes)
res8 = transforms(image=image, bboxes=bboxes)
res9 = transforms(image=image, bboxes=bboxes)
# print (res1)
# visualize_bboxes(251,'original',image,bboxes)
visualize_bboxes(252,'result1',res1['image'],res1['bboxes'])
visualize_bboxes(253,'result1',res2['image'],res2['bboxes'])
visualize_bboxes(254,'result1',res3['image'],res3['bboxes'])
visualize_bboxes(255,'result1',res4['image'],res4['bboxes'])
visualize_bboxes(256,'result1',res5['image'],res5['bboxes'])
visualize_bboxes(257,'result1',res6['image'],res6['bboxes'])
visualize_bboxes(258,'result1',res7['image'],res7['bboxes'])
visualize_bboxes(259,'result1',res8['image'],res8['bboxes'])
visualize_bboxes(251,'result1',res9['image'],res9['bboxes'])


plt.show()