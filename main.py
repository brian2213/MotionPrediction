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

    data_file_name=json.load(open("./Dataset/data_file_name.json",'r'))

    data=pickle.load(open("./Dataset/test-64data.save",'rb'))

    images=data['data']
    labels=data['label']

    print(images.shape)

    index=67

    print(data_file_name[index])

    plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
    plt.scatter(labels[index][:,1],labels[index][:,0])
    plt.show()

