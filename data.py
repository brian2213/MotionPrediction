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
from keras.utils.generic_utils import Progbar
import scipy
import pdb
import pickle

Dataset = './Dataset'
get_image_path = lambda *args: os.path.join(Dataset, *args)

labels_path = "./Dataset/annotation.json"
labels = json.load(open(labels_path, "r"))
images = list(labels.keys())


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

# show_hand_sample_image()


# dataset = h5py.File(os.path.join("/Volumes/8TB/個人文件/ics175", 'dataset.hdf5'))

def process(set,images_path,labels):
    full_images_path=[get_image_path("Color", "%s.jpg" % (image[:-2])) for image in images_path]

    # images = chain.from_iterable(load_images(full_images_path))
    images=load_images(full_images_path)

    labels = [labels[name] for name in images_path]

    # data_image = dataset.require_dataset('image/' + set, (len(labels), 256, 256, 3), dtype='float')
    # data_label = dataset.require_dataset('label/' + set, (len(labels), 21,2), dtype='float')
    # data_center = dataset.require_dataset('center/' + set, (len(labels), 2), dtype='float')

    p = Progbar(len(labels))
    index=0
    newlabels={}
    for image in images:
        label=labels[index]

        # image = convert_depth(image)
        # data_image[index], data_label[index], data_center[index] = normalize(image, label)
        newimage,newlabel,center =normalize(image, label)
        cv2.imwrite(os.path.join("/Volumes/8TB/個人文件/ics175/test", '%s.png'%(images_path[index])),newimage)

        newlabels[images_path[index]]=newlabel
        p.update(index)
        index+=1
        

    pickle.dump(newlabels, open("/Volumes/8TB/個人文件/ics175/test/annotation.save",'wb') )

def normalize(image, label):
    label = label.copy()


    # label.ix[:, 'd'] /= 10.
    # image = image / 10.

    center = label[9]

    label=np.array(label)
    label=label[:,[1,0]]

    # pdb.set_trace()
    # center.ix['d'] = label.ix[:, 'd'].median()

    bounds = bounding_box([center[1],center[0]], 525, 200).astype(int)

    image, label = clip(image, label, bounds)

    # pdb.set_trace()

    # label.ix[:, 'd'] -= center.ix['d']
    # image -= center.ix['d']

    # image = np.clip(image, -15, 15) / 15.0
    # label.ix[:, 'd'] /= 15.0
    # image, label = resize(image, label, (128, 128))
    # pdb.set_trace()
    # image = np.expand_dims(image, 0)
    # label = np.expand_dims(label, 0)
    center=[center[1],center[0]]
    # center=np.expand_dims(center,0)


    return image, label, center


# Get a bounding box of the specified dimensions in centimeters around
# a point in uvd space
def bounding_box(center, fx, size):
    bounding_box = np.array([[0, 0], [1, 1]], dtype='float')
    bounding_box -= 0.5
    bounding_box *= size
    # bounding_box *= fx / center[0]
    bounding_box += center
    # pdb.set_trace()
    return bounding_box


# Resize an image to the specified dimensions, scaling its label accordingly
def resize(image, label, dimensions):
    # label.ix[:, ['v', 'u']] *= np.array(dimensions) / image.shape[:-1]
    # plt.imshow(image)
    # plt.show()
    #
    # pdb.set_trace()
    # image = scipy.misc.imresize(np.squeeze(image), dimensions, 'bilinear', mode='F')

    return image, label


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

    # cv2.imwrite(os.path.join("/Volumes/8TB/個人文件/ics175/test", '%s.png' % ("test")), image)

    # plt.imshow(image)
    # plt.show()
    # pdb.set_trace()
    return image, label


process('train',images,labels)
# process('test')
# demo_image = dataset['image/train'][0]
# demo_label = dataset['label/train'][0]

# plt.figure(figsize=(12, 12))
# plt.imshow(demo_image.squeeze())
# plt.plot(demo_label[::3], demo_label[1::3], 'wo')
# plt.show()