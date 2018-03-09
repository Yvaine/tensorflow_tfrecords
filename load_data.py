import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import matplotlib.patches as patches

# Load training and testing datasets.
# DEPTH_FOLDER = "/Users/jing/Documents/DL/framework/tensorflow/ssd/DSOD-Tensorflow/depth_data/depth/"
# annotation_file = "/Users/jing/Documents/DL/framework/tensorflow/ssd/DSOD-Tensorflow/depth_data/track_annotations/annotation_pos_combine.txt"
DEPTH_FOLDER = "../test_image/depth_data/depth/"
annotation_file = "../test_image/depth_data/track_annotations/annotation_pos_combine.txt"

batch_size = 32
num_epoch = 10
input_shape = (300, 300)

IMAGE_TYPE_NP = np.int64
IMAGE_TYPE_TF = tf.int64

LABLE_TYPE_NP = np.float64
LABLE_TYPE_TF = tf.float64

IMAGE_SHAPE=300
LABLE_LEN = 5
IMAGE_CHANNEL = 1

def generate_filenames(depth_folder):
    filenames = os.listdir(depth_folder)
    return filenames

def read_annotation(depth_folder, raw_depth_filenames, annotaion_file):

    with open(annotaion_file) as f:
        file_lines = f.readlines()
    filenames = []
    labels = []
    for line in file_lines:
        # print("line:",line)
        if line.startswith("#") or not line:
            continue
        striped_line = line.strip("\n").split(" ")
        # print(striped_line)
        file_name = striped_line[0] + ".ppm"
        bbox = np.array([int(striped_line[2]),int(striped_line[3]),int(striped_line[4]),int(striped_line[5]), 1])


        if file_name in raw_depth_filenames:
            filenames.append(os.path.join(depth_folder, file_name))
            labels.append(bbox)

        # print(file_name)
        # print(coordinate)
    return filenames, labels

def load_data(DEPTH_FOLDER,annotation_file):
    filenames = generate_filenames(DEPTH_FOLDER)
    filenames, labels = read_annotation(DEPTH_FOLDER, filenames, annotation_file)
    images = []
    for f in filenames:
        raw_image = skimage.data.imread(f)
        images.append(raw_image)
    return np.array(images), np.array(labels)

def display_images_and_labels(images, labels, is_raw):
    """Display the first image of each label."""
    if(is_raw):
        for num in range(len(images)):
            print("display box:", labels[num][0], labels[num][1], labels[num][2],labels[num][3])
            fig0 = plt.figure(0)
            ax2 = fig0.add_subplot(111, aspect='equal')
            ax2.add_patch(
                patches.Rectangle(
                    (labels[num][0], labels[num][1]), labels[num][2],labels[num][3],fill=False # remove background
                )
            )

            image = images[num]
            _ = plt.imshow(image)
            plt.show()
    else:

        for num in range(len(images)):
            print("display box:", labels[num][0], labels[num][1], labels[num][2], labels[num][3])
            print("display image:", images[num].shape)
            fig1 = plt.figure(1)
            ax2 = fig1.add_subplot(111, aspect='equal')
            ax2.add_patch(
                patches.Rectangle(
                    (labels[num][0] * IMAGE_SHAPE - labels[num][2] * IMAGE_SHAPE / 2,
                     labels[num][1] * IMAGE_SHAPE - labels[num][3] * IMAGE_SHAPE / 2),
                    labels[num][2] * IMAGE_SHAPE,
                    labels[num][3] * IMAGE_SHAPE,
                    fill=False  # remove background
                )
            )

            image = images[num]
            _ = plt.imshow(image[:,:,0])
            plt.show()

def shuffle_data(images, labels):

    shuffle_images = []
    shuffle_labels = []

    # print (len(images))
    indices = np.arange(len(images))  # indices = the number of images in the source data set
    # print(indices)
    random.shuffle(indices)
    for i in indices:
        shuffle_images.append(images[i])
        shuffle_labels.append(labels[i])

    return shuffle_images, shuffle_labels

def sample_batch_data(images, labels, batch_size, batch_index):
    indices_begin = batch_index * batch_size
    indices_end = (batch_index+1) * batch_size
    return images[indices_begin:indices_end], labels[indices_begin:indices_end]

def calc_mean_value(images):
    mean_value = np.mean(images)
    return mean_value

def process_images(images, mean_value, input_shape, labels):

    # standard cordinate
    # [x_min, y_min, raw_width, raw_height] ===> [x_center, y_center, width, height]
    #  x_center = (x_min + raw_width/2)/image_width
    #  y_center = (y_min + raw_height/2)/image_height
    #  width = raw_width/image_width
    #  height = raw_height/image_height
    standard_label = []
    for i in np.arange(len(labels)):
        # print(images[i].shape)
        image_width = images[i].shape[1]
        image_height = images[i].shape[0]
        x_center = (labels[i][0] + labels[i][2] / 2) / image_width
        y_center = (labels[i][1] + labels[i][3] / 2) / image_height
        width = labels[i][2] / image_width
        height = labels[i][3] / image_height
        # print("box:", x_center,y_center,width,height)
        standard_label.append([x_center, y_center, width, height, labels[i][4]])

    standard_images = images - mean_value
    resize_images = []
    for raw_image in standard_images:
        resize_image = skimage.transform.resize(raw_image, (IMAGE_SHAPE,IMAGE_SHAPE))
        resize_images.append(resize_image)
    resize_images = np.array(resize_images)

    # display_images_and_labels(resize_images, labels)
    standard_images = np.expand_dims(resize_images, axis=3)
    standard_labels = np.array(standard_label)
    image_type = standard_images.dtype
    label_type = standard_labels.dtype
    print("image_type:", image_type)
    print("label_type:", label_type)
    return standard_images, standard_labels, image_type, label_type

def import_data(DEPTH_FOLDER, annotation_file):
    images, labels = load_data(DEPTH_FOLDER, annotation_file)
    # display_images_and_labels(images,labels,True)
    image_mean = calc_mean_value(images)
    images, labels, image_type, label_type = process_images(images, image_mean, input_shape, labels)
    # display_images_and_labels(images,labels,False)
    return images, labels,image_type, label_type

def encode_to_tfrecords(images, labels, tfrecord_filename):
    if os.path.exists(tfrecord_filename):
        os.remove(tfrecord_filename)
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)

    for image, label in zip(images, labels):
        # image.astype(IMAGE_TYPE_NP)
        image_raw = image.tostring()
        # print("assert:",image.shape)
        # print("assert:",label, label.shape)
        # assert(image.shape==(IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNEL))
        # assert(label.shape==(LABLE_LEN,))
        # label.astype(LABLE_TYPE_NP)
        label_raw = label.tostring()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw]))
                    # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }
            )
        )
        serialized = example.SerializeToString()  # 序列化的写入
        writer.write(serialized)  # 这里相当于把序列化的数据已经写入了'test.tfrecord'文件中
    print ('writer DOWN!')
    writer.close()

def decode_from_tfrecords(tfrecord_filename,is_batch, batch_size, image_type, label_type):
    # output file name string to a queue
    filename_queue = tf.train.string_input_producer([tfrecord_filename], num_epochs=None)
    # create a reader from file queue，建立一个阅读器
    reader = tf.TFRecordReader()
    _, serialzed_example = reader.read(filename_queue)  # 从阅读器中读取数据

    features = tf.parse_single_example(serialzed_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       }
                                       )
    image_out = features['image']
    label_out = features['label']
    image = tf.decode_raw(image_out, image_type)
    image = tf.reshape(image, [IMAGE_SHAPE,IMAGE_SHAPE,IMAGE_CHANNEL])

    label = tf.decode_raw(label_out, label_type)
    label = tf.reshape(label, [LABLE_LEN,])
    # label = tf.cast(features['label'], LABLE_TYPE_TF)

    if is_batch:
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size
        image, label = tf.train.shuffle_batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=3,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    return image, label

def main1():
    with tf.Graph().as_default():

        images, labels = load_data(DEPTH_FOLDER, annotation_file)
        print("raw_images shape:", images.shape)
        num_iterators = num_epoch * batch_size

        # display_images_and_labels(images, labels)

        image_mean = calc_mean_value(images)
        #
        images, labels = process_images(images, image_mean, input_shape, labels)
        #
        print("expand images shape:", images.shape)
        # display_images_and_labels(images, labels)
        # print(images.shape)
        # print(labels.shape)
        # print(image_mean)

        with tf.Session() as sess:

            for i in range(num_epoch):
                shuffle_image, shuffle_label = shuffle_data(images, labels)
                batch_num = int(len(shuffle_image) / batch_size)

                for batch_indice in range(batch_num):
                    image_batch, label_batch = sample_batch_data(shuffle_image, shuffle_label, batch_size, batch_indice)

                    # print("[%d]epoch[%d]batch[%d]iter" %(i, batch_indice, i * batch_num + batch_indice))
                    # print(np.array(image_batch).shape, np.array(label_batch).shape)

        # with tf.Session() as sess:
        #     for i in range(num_iterators):
        # print(sess.run(image))
        # sess.run(label)

def main2():
    # step1: fetch all data
    # train, test = tf.keras.datasets.mnist.load_data()
    # print(train)
    # mnist_x, mnist_y = train
    # print(mnist_x.shape)
    # mnist_ds = tf.data.Dataset.from_tensor_slices(mnist_x)
    # print(mnist_ds)
    images, labels = import_data(DEPTH_FOLDER, annotation_file)
    print("images shape:", images.shape)
    image_ds = tf.data.Dataset.from_tensor_slices(images)
    print("image_ds shape:", image_ds)

def main3():
    images, labels, image_type, label_type = import_data(DEPTH_FOLDER, annotation_file)
    # print("images shape:", images.shape)

    tf_records = "depth.records"
    encode_to_tfrecords(images,labels,tf_records)
    image, label = decode_from_tfrecords(tf_records,True,batch_size,image_type, label_type)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        _ = sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1):
            image_batch, label_batch = sess.run([image, label])
            # img_1 = tf.reshape(image_batch[0, :, :, :], [IMAGE_HEIGHT, IMAGE_WIDTH])
            print (image_batch.shape)
            print (image_batch[0, :, :,0].shape)

            for num in range(batch_size):
                fig2 = plt.figure(1)
                ax2 = fig2.add_subplot(111, aspect='equal')
                ax2.add_patch(
                    patches.Rectangle(
                        (label_batch[num][0] * IMAGE_SHAPE - label_batch[num][2] * IMAGE_SHAPE / 2,
                         label_batch[num][1] * IMAGE_SHAPE - label_batch[num][3] * IMAGE_SHAPE / 2),
                        label_batch[num][2] * IMAGE_SHAPE,
                        label_batch[num][3] * IMAGE_SHAPE,
                        fill=False  # remove background
                    )
                )
                plt.imshow(image_batch[num, :, :, 0])
                plt.show()

if __name__ == '__main__':
    # main1()
    # main2()
    main3()