import tensorflow as tf
import json
if __name__ == '__main__':
    Joints = json.load(open("./Dataset/annotation.json", "r"))
    names = Joints.keys()
    print(len(Joints[list(names)[0]]))
