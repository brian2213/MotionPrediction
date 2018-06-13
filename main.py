import tensorflow as tf
import json
if __name__ == '__main__':
    Joints = json.load(open("./Dataset/annotation.json", "r"))
    names = Joints.keys()
    print(len(Joints[list(names)[0]]))

    #
    # image_contents = tf.read_file("./Dataset/Color/" + "000_251" + ".jpg")
    # image = tf.image.decode_jpeg(image_contents,channels=3)
    #
    # resized_image = tf.image.resize_images(image, [1920, 1080])
    # print(resized_image)

#
#     filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./Dataset/*/*.jpg"))
#     # filename_queue = tf.train.string_input_producer(["./Dataset/Color/" + "000_251" + ".jpg"])
#     image_reader = tf.WholeFileReader()
#     _, image_file = image_reader.read(filename_queue)
#     image = tf.image.decode_jpeg(image_file)
#     init = (tf.global_variables_initializer(), tf.local_variables_initializer())
#
# with tf.Session() as sess:
#     # Required to get the filename matching to run.
#     sess.run(init)
#
#     # Coordinate the loading of image files.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     # Get an image tensor and print its value.
#     image_tensor = sess.run([image])
#     print(image_tensor)
#
#     # Finish off the filename queue coordinator.
#     coord.request_stop()
#     coord.join(threads)