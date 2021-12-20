import numpy as np
import albumentations as A
import random
import cv2 
import matplotlib.pyplot as plt
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def draw_bboxes(img, bboxes):
    height, width = img.shape[:2]
    for x, y, w, h, label in bboxes:
        class_name = category_id_to_name[label]
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

random.seed(0)
np.random.seed(0)
image = plt.imread('./IMG_1393.JPG')
bboxes = [[0.278081,0.547082,0.038222,0.049072,0]
,[0.441108, 0.511273, 0.038222, 0.057029,8]
,[0.505070, 0.511273, 0.041342, 0.057029,18]
,[0.567473, 0.509947, 0.039782, 0.059682,20]
,[0.631825, 0.509947, 0.042122, 0.059682,22]
,[0.632995, 0.377321, 0.041342, 0.054377,24]
,[0.697738, 0.377984, 0.041342, 0.058355,26]
,[0.764431, 0.376658, 0.045242, 0.061008,28]]
# category_id_to_name = {0:'open',8:'B1F'}
category_id_to_name = {
0:'open',  
1:'open_on',
2:'close',
3:'close_on',
4:'up',
5:'down',
6:'up_on',
7:'down_on',
8:'B1F',
9:'B1F_on',
10:'B2F',
11:'B2F_on',
12:'B3F',
13:'B3F_on',
14:'B4F',
15:'B4F_on',
16:'B5F',
17:'B5F_on',
18:'1F',
19:'1F_on',
20:'2F',
21:'2F_on',
22:'3F',
23:'3F_on',
24:'4F',
25:'4F_on',
26:'5F',
27:'5F_on',
28:'6F',
29:'6F_on',
30:'7F',
31:'7F_on',
32:'8F',
33:'8F_on',
34:'9F',
35:'9F_on',
36:'10F',
37:'10F_on',
38:'11F',
39:'11F_on',
40:'12F',
41:'12F_on',
42:'13F',
43:'13F_on',
44:'14F',
45:'14F_on',
46:'15F',
47:'15F_on',
48:'16F',
49:'16F_on',
50:'17F',
51:'17F_on',
52:'18F',
53:'18F_on',
54:'19F',
55:'19F_on',
56:'20F',
57:'20F_on',
58:'21F',
59:'21F_on',
60:'22F',
61:'22F_on',
62:'23F',
63:'23F_on',
64:'24F',
65:'24F_on',
66:'25F',
67:'25F_on',
68:'26F',
69:'26F_on',
70:'27F',
71:'27F_on',
72:'28F',
73:'28F_on',
74:'29F',
75:'29F_on',
76:'30F',
77:'30F_on',
78:'person',
79:'bicycle',
80:'car',
81:'motorcycle',
82:'truck',
83:'dog',
84:'gate1'}

# transforms = A.Compose([A.Resize(416, 416)], bbox_params=A.BboxParams(format="yolo"))
transforms = A.Compose([
    # A.Resize(416, 416),
    # A.RandomCrop(width=256, height=256),
    A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1,p=0.9)],
bbox_params=A.BboxParams(format="yolo"))


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