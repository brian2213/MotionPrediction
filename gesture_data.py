import json
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import json
from itertools import chain
import pandas as pd
from PIL import Image
#from keras.utils.generic_utils import Progbar
import scipy
import pdb
import pickle
import copy

Dataset = './Dataset'
get_image_path = lambda *args: os.path.join(Dataset, *args)

path2 = "./Dataset/gesture_labeled.json"
gesture_dict = json.load(open(path2, "r"))

labels_path = "./Dataset/annotation.json"
labels = json.load(open(labels_path, "r"))
labels2 = {}
for g_label in gesture_dict:
    if labels.has_key(g_label):
        labels2[g_label] = labels[g_label]
images = list(labels2.keys())


#Changed by hcz



def load_images(images, dtype='uint8'):
    for image in images:
        yield np.asarray(cv2.imread(image), dtype)


def show_demo_image():
    demo_images = load_images(
        [get_image_path(data_type, "%s.jpg" % (image[:-2])) for data_type in ["Color", "Depth"] for image in
         images[:3]])

    plt.figure(figsize=(16, 12))
    for i, demo_image in enumerate(demo_images):
        plt.subplot(231 + i)
        ax = plt.gca()
        ax.set_axis_off()
        ax.set_xlim([0, demo_image.shape[1]])
        ax.set_ylim([demo_image.shape[0], 0])
        RGB_img = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)
        plt.imshow(RGB_img)
    plt.show()


def show_hand_sample_image():
    demo_image = cv2.imread(get_image_path("Color", "%s.jpg" % (images[0][:-2])))
    demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)

    demo_label = labels[images[0]]

    plt.figure(figsize=(12, 12))
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.scatter([demo_label[i][0] for i in range(len(demo_label))],[demo_label[i][1] for i in range(len(demo_label))])
    plt.gca().set_autoscale_on(False)
    plt.gca().set_axis_off()
    plt.imshow(demo_image.squeeze())
    plt.show()




def process(set,images_path,labels,gesture_dict,image_res=32):
    full_images_path=[get_image_path("Color", "%s.jpg" % (image[:-2])) for image in images_path]
    images=load_images(full_images_path)

    new_labels = [labels[name] for name in images_path]
    labels_name = [name for name in images_path]
    # data_image = dataset.require_dataset('image/' + set, (len(labels), 256, 256, 3), dtype='float')
    # data_label = dataset.require_dataset('label/' + set, (len(labels), 21,2), dtype='float')
    # data_center = dataset.require_dataset('center/' + set, (len(labels), 2), dtype='float')

    #p = Progbar(len(labels))
    index=0
    newlabels={}

    # with open('data.txt', 'w') as outfile:
    #     json.dump(images_path, outfile)
    # pdb.set_trace()

    data=np.empty((len(new_labels),image_res,image_res,3),'uint8')
    target=np.empty((len(new_labels),21,2),'float')
    image_center=np.empty((len(new_labels),2),'float')
    gestures = np.empty((len(new_labels),1),'int')

    for image in images:
        label=new_labels[index]
        newimage,newlabel,center =normalize(image, label)
        gesture = gesture_dict[labels_name[index]]
        print gesture
        newimage1, newlabel1=resize(newimage,newlabel,(image_res,image_res))


        data[index]=newimage1
        target[index]=newlabel1
        image_center[index]=center
        gestures[index]=gesture


        #p.update(index)
        index+=1

    datas={}
    datas['data']=data
    datas['label']=target
    datas['center']=image_center
    datas['gesture']=gestures
    pickle.dump(datas, open(os.path.join(Dataset, "test-" + str(image_res) + "data.save"), 'wb'))



def normalize(image, label):
    label = copy.deepcopy(label)

    center = label[9]
    label=np.array(label)
    label=label[:,[1,0]]

    bounds = bounding_box([center[1],center[0]], 525, 192).astype(int)

    image, label = clip(image, label, bounds)
    center=[center[1],center[0]]

    return image, label, center


# Get a bounding box of the specified dimensions in centimeters around
# a point in uvd space
def bounding_box(center, fx, size):
    bounding_box = np.array([[0, 0], [1, 1]], dtype='float')
    bounding_box -= 0.5
    bounding_box *= size
    bounding_box += center
    return bounding_box


# Clip an image to the specified bounding box, translating its label accordingly
# Bounding box should look like np.array([[x_1, y_1], [x_2, y_2]]), where
# (x_1, y_1) are the coordinates of the lower left corner and
# (x_2, y_2) are the coordinates of the upper right corner
def clip(image, label, bounding_box):
    # label.ix[:, ['v', 'u']] -= bounding_box[0]

    label-=bounding_box[0]
    image_box = np.array([[0, 0], image.shape[:-1]], dtype='int')

    padding = np.array([image_box[0] - bounding_box[1], bounding_box[0] - image_box[1]]).clip(0)

    bounding_box += padding[0]

    padding = np.concatenate((padding.T, np.array([[0, 0]])))

    image = np.pad(image, padding, 'edge')
    image = image[slice(*bounding_box[:, 0]), slice(*bounding_box[:, 1])]

    return image, label
def resize(image,label,dimension):
    resized_image = cv2.resize(image, dimension)
    label *= image.shape[0]/dimension[0]
    return resized_image, label

# process('train',images,labels,image_res=64)
process('train',images,labels2,gesture_dict)