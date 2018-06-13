import tensorflow as tf


def my_model(X, y, is_training):
    Wconv1 = tf.get_variable("Wconv1", shape=[2, 2, 3, 512])
    bconv1 = tf.get_variable("bconv1", shape=[512])
    Wconv2 = tf.get_variable("Wconv2", shape=[2, 2, 512, 256])
    bconv2 = tf.get_variable("bconv2", shape=[256])
    Wconv3 = tf.get_variable("Wconv3", shape=[2, 2, 256, 128])
    bconv3 = tf.get_variable("bconv3", shape=[128])
    #   Wconv4 = tf.get_variable("Wconv4",shape=[2,2,128,64])
    #   bconv4 = tf.get_variable("bconv4", shape=[64])

    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 1, 1, 1], padding="SAME") + bconv1
    h1 = tf.nn.relu(a1)
    h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=is_training, scope='bn1')

    ph2 = tf.layers.max_pooling2d(inputs=h2, pool_size=[2, 2], strides=2)

    a2 = tf.nn.conv2d(ph2, Wconv2, strides=[1, 1, 1, 1], padding="SAME") + bconv2
    h2 = tf.nn.relu(a2)
    h3 = tf.contrib.layers.batch_norm(h2, center=True, scale=True, is_training=is_training, scope='bn2')

    ph3 = tf.layers.max_pooling2d(inputs=h3, pool_size=[2, 2], strides=2)

    a3 = tf.nn.conv2d(ph3, Wconv3, strides=[1, 1, 1, 1], padding="SAME") + bconv3
    h3 = tf.nn.relu(a3)
    h4 = tf.contrib.layers.batch_norm(h3, center=True, scale=True, is_training=is_training, scope='bn3')

    ph4 = tf.layers.max_pooling2d(inputs=h4, pool_size=[2, 2], strides=2)
    ph4 = tf.layers.dropout(ph4, 0.5)

    W1 = tf.get_variable("W1", shape=[2048, 512])
    b1 = tf.get_variable("b1", shape=[512])
    W2 = tf.get_variable("W2", shape=[512, 128])
    b2 = tf.get_variable("b2", shape=[128])
    W3 = tf.get_variable("W3", shape=[128, 21])
    b3 = tf.get_variable("b3", shape=[21])

    ph4_flat = tf.reshape(ph4, [-1, 2048])
    h5 = tf.matmul(ph4_flat, W1) + b1
    h6 = tf.nn.relu(h5)

    h7 = tf.reshape(h6, [-1, 512])
    h8 = tf.matmul(h7, W2) + b2
    h9 = tf.nn.relu(h8)

    h10 = tf.reshape(h9, [-1, 128])
    h11 = tf.matmul(h10, W3) + b3
    #   h12 = tf.nn.relu(h11)

    return h11


tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

y_out = my_model(X, y, is_training)
total_loss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4)  # select optimizer and set learning rate


# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)
