import tensorflow as tf
import json
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb

if __name__ == '__main__':
    # Joints = json.load(open("./Dataset/annotation.json", "r"))
    # names = Joints.keys()
    # print(len(Joints[list(names)[0]]))
    data=pickle.load(open("./Dataset/test-64data.save",'rb'))

    images=data['data']
    labels=data['label']

    print(images.shape)

    index=4

    plt.imshow(images[index])
    plt.scatter(labels[index][:,1],labels[index][:,0])
    plt.show()

