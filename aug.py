import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White
bb_image = None

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_center,y_center,width,height = bbox
    # x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    # bb_image = img
    # return bb_image
    
    print('done')

image = plt.imread('./IMG_1393.JPG')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes = [[5.66, 138.95, 147.09, 164.88], [366.7, 80.84, 132.8, 181.84]]
# bboxes = [[0.278081,0.547082,0.038222,0.049072],[0.441108, 0.511273, 0.038222, 0.057029]]
category_ids = [0,8]
category_id_to_name = {0:'open',8:'B1F'}

# transform = A.Compose([
#     A.HorizontalFlip(p=1),
# ], bbox_params=A.BboxParams(format='yolo',min_visibility=0.4, label_fields=[]))

# transformed = transform(image=image, bboxes=bbox)
# transformed_image = transformed['image']
# transformed_bboxes = transformed['bboxes']


# plt.imshow(transformed_image)

# # bounindg box
# for transformed_bbox in transformed_bboxes:
#     xmin, ymin, xmax, ymax, category = transformed_bbox
#     rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='red', fill=False)
#     ax = plt.gca()
#     ax.add_patch(rect)

# plt.xticks([]); plt.yticks([])
# plt.show()

visualize(image, bboxes, category_ids, category_id_to_name)
# print (image)

plt.show()
