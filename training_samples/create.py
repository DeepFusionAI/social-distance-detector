#@title Download the COCO `train2014` training annotations and subsample 200 images for person only
#@markdown **Note** - Run this if you want to create your own dataset.
from pycocotools.coco import COCO
import requests
import random

# instantiate COCO specifying the annotations json path
coco = COCO('annotations_trainval2014/annotations/instances_train2014.json')
# Specify a list of category names of interest
catIds = coco.getCatIds(catNms=['person'])
# Get the corresponding image ids and images using loadImgs
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)

random.shuffle(images)

two_hundred_images = images[:200]



for im in two_hundred_images:
    img_data = requests.get(im['coco_url']).content
    with open('/content/train_samples/' + im['file_name'], 'wb') as handler:
        handler.write(img_data)