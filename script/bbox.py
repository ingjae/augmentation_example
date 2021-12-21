import numpy as np
import albumentations as A
import random
import cv2 
import matplotlib.pyplot as plt
import names
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

image = plt.imread('../dataset/IMG_1393.JPG')
bboxes = [[0.278081,0.547082,0.038222,0.049072,0],
        [0.441108, 0.511273, 0.038222, 0.057029,8],
        [0.505070, 0.511273, 0.041342, 0.057029,18],
        [0.567473, 0.509947, 0.039782, 0.059682,20],
        [0.631825, 0.509947, 0.042122, 0.059682,22],
        [0.632995, 0.377321, 0.041342, 0.054377,24],
        [0.697738, 0.377984, 0.041342, 0.058355,26],
        [0.764431, 0.376658, 0.045242, 0.061008,28]]


random.seed(0)
np.random.seed(0)

transforms = A.Compose([
    # A.Resize(416, 416),
    # A.RandomCrop(width=256, height=256),
    A.RandomSizedBBoxSafeCrop(height = 600,width = 800,erosion_rate =  0.0, interpolation = 1,p=1),
    # A.RandomSizedBBoxSafeCrop(height = 60,width = 80,erosion_rate =  0.0, interpolation = 1,p=1),
    # A.UnsharpMask (blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.2, 0.5), threshold=10, always_apply=False, p=1),
    A.RandomToneCurve (scale=0.5, always_apply=False, p=1),
    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5,p=0.9)
    ],
bbox_params=A.BboxParams(format="yolo"))

def visualize_bboxes():
    pass
res1 = transforms(image=image, bboxes=bboxes)
res2 = transforms(image=image, bboxes=bboxes)
res3 = transforms(image=image, bboxes=bboxes)
res4 = transforms(image=image, bboxes=bboxes)
# print (res)

plt.subplot(241, title="original")
draw_bboxes(image, bboxes)
plt.imshow(image, vmin=0, vmax=255)

plt.subplot(245, title="result1")
draw_bboxes(res1["image"], res1["bboxes"])
plt.imshow(res1["image"], vmin=0, vmax=255)

plt.subplot(246, title="result2")
draw_bboxes(res2["image"], res2["bboxes"])
plt.imshow(res2["image"], vmin=0, vmax=255)

plt.subplot(247, title="result3")
draw_bboxes(res3["image"], res3["bboxes"])
plt.imshow(res3["image"], vmin=0, vmax=255)

plt.subplot(248, title="result4")
draw_bboxes(res4["image"], res4["bboxes"])
plt.imshow(res4["image"], vmin=0, vmax=255)


plt.show()