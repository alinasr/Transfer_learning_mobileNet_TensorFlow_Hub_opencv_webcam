# -*- coding: utf-8 -*-
"""
Created on Wed May  2 20:26:55 2018

@author: Ali Nasr
"""

# import important
import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import os, os.path
from random import shuffle, sample
import re

data_path = "/home/ali/PycharmProjects/tensorHub/data/"
label_zero_path = data_path + "label_zero/"
label_one_path = data_path + "label_one/"



#height_img, width_img = hub.get_expected_image_size(m)
#depth = hub.get_num_image_channels(m)

height_img = 128
width_img = 128


width_img_cam = 352
height_img_cam = 288

frame = np.zeros((height_img, width_img, 1), np.uint8)
windowName = 'cam'
cv2.namedWindow(windowName)
l = 0

hm_epochs = 100
number_of_data = 10
n_classes = 2
batch_size = 10



def run_and_save_bottleneck(input_data, batch_step):


# conevrs images from unin8 to float numbers
    image_as_float = tf.image.convert_image_dtype(input_data, tf.float32)



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        decoded_images = sess.run(image_as_float)

# starting a sessin tu get bottleneck values
    bottleneck_tensor = m(x);


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        x_epoch = decoded_images[0: batch_size]
        bottleneck_value = sess.run(bottleneck_tensor, feed_dict={x: x_epoch})

        for batch in range(1, batch_step):
            x_epoch = decoded_images[batch  * batch_size : (batch + 1) * batch_size]
            batche_value = sess.run(bottleneck_tensor, feed_dict={x:x_epoch})
            bottleneck_value = np.concatenate((bottleneck_value, batche_value))
            print(bottleneck_value.shape)
    print("finishing up")


    return bottleneck_value




def mobileNet(bottleneck_plcaholder):


    weights = {'out': tf.Variable(tf.random_normal([1024, n_classes], stddev=0.001))}

    biases = {'out': tf.Variable(tf.random_normal([n_classes], stddev=0.001))}


    #fc = tf.nn.relu(tf.matmul(fc, weights['Weights_FC']) + biases['Biase_FC'])

    # fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(bottleneck_plcaholder, weights['out']) + biases['out']

    return  output






def train_neural_network(bottleneck_plcaholder, bottleneck_valus, labels, batch_step):

    prediction = mobileNet(bottleneck_plcaholder)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for batch in range(batch_step):
                epoch_x = bottleneck_valus[batch * batch_size: (batch + 1) * batch_size]
                epoch_y = labels[batch * batch_size: (batch + 1) * batch_size]

                _, c = sess.run([optimizer, cost], feed_dict={bottleneck_plcaholder: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        # please customize the directory for your project
        saver.save(sess, '/home/ali/PycharmProjects/tensorHub/save/my_test_model')

        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accyracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


def feed_for():
    global l
    #img = np.zeros((number_of_data, height_img, width_img))
    #label = np.zeros(number_of_data).astype(int)
    i = 0

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    cap.set(3, width_img_cam);
    cap.set(4, height_img_cam);

    print(cap.get(3))
    print(cap.get(4))

    sess = tf.Session()
    # please customize the directory for your project
    saver = tf.train.import_meta_graph('/home/ali/PycharmProjects/test1/saved/my_test_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/home/ali/PycharmProjects/test1/saved/./'))

    graph = tf.get_default_graph()
    # w1 = graph.get_tensor_by_name("w1:0")
    # w2 = graph.get_tensor_by_name("w2:0")

    # Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("add_1:0")
    forward = tf.nn.softmax(logits=op_to_restore)

    while (True):
        ret, camera_img = cap.read()
        camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(camera_img, (width_img, height_img))
        cv2.imshow(windowName, frame)

        epoch_x = frame.reshape(1, width_img * height_img)
        feed_dict = {x: epoch_x}

        sess.run(tf.global_variables_initializer())
        # print(forward.eval(feed_dict))
        print(sess.run(op_to_restore, feed_dict))

        if cv2.waitKey(1) == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
    cap.release()





def mouse(event, x, y, flags, param):
    global l

    if event == cv2.EVENT_MBUTTONDOWN:
        l = 1  # it actually zero it will be changed in capture_ dataset function
    if event == cv2.EVENT_LBUTTONDOWN:
        l = 2  # it actually 1 it will be changed in capture_ dataset function


# bind the callback function to window
cv2.setMouseCallback(windowName, mouse)


def load_data_from_files():
    pattern = r"label_one"

    extensions = ['jpg', 'png']
    file_list_ones = []
    file_list_zeros = []




    for extension in extensions:
        file_glob_ones = os.path.join(label_one_path, label_one_path, '*.' + extension)
        file_glob_zeros = os.path.join(label_zero_path, label_zero_path, '*.' + extension)
        file_list_zeros.extend(tf.gfile.Glob(file_glob_zeros))
        file_list_ones.extend(tf.gfile.Glob(file_glob_ones))

    files_one_and_zero = file_list_zeros + file_list_ones
    all_files = sample(files_one_and_zero, len(files_one_and_zero));
    input_data = np.zeros((len(all_files), height_img, width_img, 3), np.uint8)
    labels = np.zeros((len(all_files), 1) , np.uint8)


    for file in all_files:
        input_data[all_files.index(file)] = cv2.imread(file, -1)
        if re.search(pattern, file):
            labels[all_files.index(file)] = 1
        else:
            labels[all_files.index(file)] = 0


    print("number of train data is {}".format(len(labels)))
    labels = np.eye(n_classes)[labels.reshape(-1)]
    return input_data, labels



def capture_and_save_dataset():
    global l
    img = np.zeros((number_of_data, height_img, width_img, 3))
    label = np.zeros(number_of_data).astype(int)
    i = 0

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    cap.set(3, width_img_cam);
    cap.set(4, height_img_cam);

    print(cap.get(3))
    print(cap.get(4))

    while (True):
        ret, camera_img = cap.read()
        #camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(camera_img, (width_img, height_img))
        cv2.imshow(windowName, frame)

        if l == 1:
            label[i] = 0;
            img[i] = frame
            l = 0
            i += 1
        elif l == 2:
            label[i] = 1
            img[i] = frame
            l = 0
            i += 1

        if i == number_of_data:
            break

        if cv2.waitKey(1) == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
    cap.release()

    files_count_in_label_zero = len([f for f in os.listdir(label_zero_path)
                                     if os.path.isfile(os.path.join(label_zero_path, f))])
    files_count_in_label_one = len([f for f in os.listdir(label_one_path)
                                    if os.path.isfile(os.path.join(label_zero_path, f))])

    files_counts = files_count_in_label_one + files_count_in_label_zero

    for j in label:
        if j == 0:
            files_counts += 1
            cv2.imwrite(label_zero_path + "{}.jpg".format(files_counts), img[j],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if j == 1:
            files_counts += 1
            cv2.imwrite(label_one_path + "{}.jpg".format(files_counts), img[j],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if cv2.waitKey(1) == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
    label = np.eye(n_classes)[label.reshape(-1)]
    return img, label





def main():
    #images1, labels1 = capture_and_save_dataset()

    input_data, labels = load_data_from_files()

    number_of_data, _,_,_ = input_data.shape
    batch_step = int(number_of_data/batch_size)

    bottleneck_valus = run_and_save_bottleneck(input_data, batch_step)

    train_neural_network(bottleneck_plcaholder, bottleneck_valus, labels, batch_step)


main()



