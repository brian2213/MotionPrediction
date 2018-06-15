import json
import pickle

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import *
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import Adam
import pdb

np.random.seed(1234)
def get_image_data(num_training=63000, num_validation=1000, num_test=10000):
    """
    Load the image dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw image data
    image_dir = "./Dataset/test-32data.save"

    data = pickle.load(open(image_dir, 'rb'))
    X_train, y_train, X_test, y_test = data['data'][:80000], data['label'][:80000], data['data'][80000:], data['label'][
                                                                                                          80000:]
    # X_train = np.array(X_train, 'float64')
    # y_train = np.array(y_train, 'float64')
    # X_test = np.array(X_test, 'float64')
    # y_test = np.array(y_test, 'float64')
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    # mean_image = np.mean(X_train, axis=0)
    # X_train -= mean_image
    # X_val -= mean_image
    # X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


def simple_model(X, y):
    # define our weights (e.g. init_two_layer_convnet)

    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 21, 2])
    b1 = tf.get_variable("b1", shape=[21, 2])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 2, 2, 1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1, [-1, 5408])
    y_out = tf.matmul(h1_flat, W1) + b1
    return y_out


def keras_model():
    pass


def main():
    # Invoke the above function to get our data.
    X_train, y_train, X_val, y_val, X_test, y_test = get_image_data()
    y_train = y_train.reshape(-1, 42)
    y_val = y_val.reshape(-1, 42)
    y_test = y_test.reshape(-1, 42)
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    model = Sequential([
        Convolution2D(
            nb_filter=12,
            nb_row=5,
            nb_col=5,
            subsample=(2, 2),
            input_shape=(32, 32, 3)
        ),
        LeakyReLU(
            alpha=0.05
        ),
        Convolution2D(
            nb_filter=12,
            nb_row=5,
            nb_col=5,
            subsample=(2, 2),
        ),
        LeakyReLU(
            alpha=0.05
        ),
        Convolution2D(
            nb_filter=12,
            nb_row=5,
            nb_col=5,
        ),
        LeakyReLU(
            alpha=0.05
        ),
        Flatten(),
        Dense(
            output_dim=1024,
            activation='relu'
        ),
        Dense(
            output_dim=1024,
            activation='relu'
        ),
        Dense(
            output_dim=21,
        ),
        Dense(
            output_dim=42,
            # weights=(pca_eigenvectors, pca_mean),
            trainable=False
        )
    ])

    model.compile(
        optimizer=Adam(),
        loss='mse'
    )

    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    model.fit(
        X_train,
        y_train,
        batch_size=100,
        epochs=1,
    )

    # pickle.dump(model,open("keras.save",'wb'))
    # json.dump(model,open("keras.save", 'wb'))
    model_json = model.to_json()
    with open("keras.save", "w") as json_file:
        json_file.write(model_json)


def showModel(num_training=49000, num_validation=1000, num_test=10000):
    image_dir = "./Dataset/test-32data.save"
    data = pickle.load(open(image_dir, 'rb'))
    X_train, y_train, X_test, y_test = data['data'][:80000], data['label'][:80000], data['data'][80000:], data['label'][
                                                                                                          80000:]
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # model=pickle.load(open("keras.save",'r'))
    # model=json.load(open("keras.save", 'r'))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    json_file = open('keras.save', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    demo = X_train[1]

    sample = np.array(X_train, 'float64')
    mean_image = np.mean(sample, axis=0)
    sample -= mean_image
    predictions_train = model.predict(sample).reshape(-1, 21, 2)

    sample = np.array(X_val, 'float64')
    mean_image = np.mean(sample, axis=0)
    sample -= mean_image
    predictions_val = model.predict(sample).reshape(-1, 21, 2)


    print("train set accuracy: %s"%(calculate_error(y_train,predictions_train)))
    print("train set accuracy: %s" % (calculate_error(y_val, predictions_val)))

    out = predictions_train[1]
    # print(out.shape)
    # pdb.set_trace()
    plt.imshow(X_train[1])
    plt.scatter(out[:, 1], out[:, 0])
    plt.show()

def calculate_error(label1,label2):

    square_error = np.square(label1 - label2)

    error_sum=np.sum(square_error)/square_error.size

    return np.sqrt(error_sum)



def test_data():
    # Joints = json.load(open("./Dataset/annotation.json", "r"))
    # names = Joints.keys()
    # print(len(Joints[list(names)[0]]))

    data_file_name = json.load(open("./Dataset/data_file_name.json", 'r'))

    data = pickle.load(open("./Dataset/test-32data.save", 'rb'))

    images = data['data']
    labels = data['label']

    print(images.shape)
    print(labels.shape)

    index = data_file_name.index('089_2441_L')
    print(index)
    print(data_file_name[81015])

    plt.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
    # plt.imshow(images[index])
    plt.scatter(labels[index][:, 1], labels[index][:, 0])
    plt.show()


if __name__ == '__main__':
    test_data()
    # main()
    # showModel()
