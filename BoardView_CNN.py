import time
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import tensorflow as tf

# def max_pool_2x2(x):
# return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    if name:
        return tf.Variable(initial, name)
    else:
        return tf.Variable(initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    if name:
        return tf.Variable(initial, name)
    else:
        return tf.Variable(initial)


# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


# PIC_HEIGHT = 64
# PIC_WIDTH = 64
PIC_HEIGHT = 384
PIC_WIDTH = 216

def make_char_dict():
    char_dict = {}
    for i, c in enumerate(CHARS):
        char_dict[c] = i
    return char_dict

DIR_NAME = './generated_data/'
CHARS = '0KQRBNPkqrbnp'
CHARS_LENGTH = len(CHARS)
CHAR_DICT = make_char_dict()

class Network:
    NETWORK_FILE = "./view.ckpt"

    def __init__(self, value_net=True):
        self.net_file = Network.NETWORK_FILE
        self.sess = tf.Session()
        self.assemble_network()

    def layer_conv(self, input_layer, num_in, num_out, padding, stride=4):
        W_conv = weight_variable([padding, padding, num_in, num_out])
        b_conv = bias_variable([num_out])

        convolve = conv2d(input_layer, W_conv, stride) + b_conv
        ################################################
        convolve = tf.layers.batch_normalization(convolve)
        ################################################
        return convolve

    def layer_fully_connected(self, layer_flatten, dim_in, dim_out):
        W_fc = weight_variable([PIC_HEIGHT * PIC_WIDTH * dim_in, dim_out])
        b_fc = bias_variable([dim_out])

        matmul_fc = tf.matmul(layer_flatten, W_fc) + b_fc
        output_layer = tf.nn.relu(matmul_fc)
        return output_layer

    def layer_output(self, layer_flatten, dim_in):
        W_fc = weight_variable([dim_in, PIC_HEIGHT * PIC_WIDTH * CHARS_LENGTH])
        b_fc = bias_variable([PIC_HEIGHT * PIC_WIDTH * CHARS_LENGTH])

        self.matmul_fc = tf.matmul(layer_flatten, W_fc) + b_fc
        ################################################
        # batch_norm = tf.layers.batch_normalization(self.matmul_fc)
        # output_layer = tf.nn.sigmoid(batch_norm)
        ################################################
        output_layer = tf.nn.sigmoid(self.matmul_fc)
        return output_layer

    def assemble_network(self):

        self.x = tf.placeholder(
            tf.float32, shape=[None, PIC_HEIGHT * PIC_WIDTH], name="x_data")
        x_image = tf.reshape(self.x, [-1, PIC_HEIGHT, PIC_WIDTH, 1])

        ################################################
        # self.y_ = tf.placeholder(
        #     tf.float32, shape=[None, 8 * 8 * CHARS_LENGTH], name="y_data")#, CHARS_LENGTH])
        ################################################
        self.y_ = tf.placeholder(tf.int32, shape=[None, 8 * 8], name="y_data")

        input_dim = 1
        output_dim = 8
        conv1 = self.layer_conv(x_image, input_dim, output_dim, padding=11, stride=4)
        conv1_layer = tf.nn.relu(conv1)

        input_dim = output_dim
        output_dim *= 2
        conv2 = self.layer_conv(conv1_layer, input_dim, output_dim, padding=5, stride=3)
        conv2_layer = tf.nn.relu(conv2)

        input_dim = output_dim
        output_dim *= 2
        conv3 = self.layer_conv(conv2_layer, input_dim, output_dim, padding=3, stride=2)
        conv3_layer = tf.nn.relu(conv3)

        """
        for _ in range(3):
            temp_layer = self.layer_conv(conv_layer, output_dim, output_dim)
            temp_layer = tf.nn.relu(temp_layer)
            conv_layer = self.layer_conv(
                temp_layer, output_dim, output_dim) + conv_layer
            conv_layer = tf.nn.relu(conv_layer)

        """
        flatten = tf.reshape(conv3_layer, [-1, 16 * 9 * output_dim])
        # output_layer = self.layer_output(
        #     flatten, ROW_DIM * COLUMN_DIM * output_dim)

        W_fc = weight_variable([16 * 9 * output_dim, 8 * 8 * CHARS_LENGTH], 'W')
        b_fc = bias_variable([8 * 8 * CHARS_LENGTH], 'b')
        matmul_fc = tf.add(tf.matmul(flatten, W_fc), b_fc, name = 'matmul')

        information = tf.reshape(matmul_fc, [-1, 8*8, CHARS_LENGTH])
        self.inform = tf.argmax(information, axis = 2)

        ###############################################
        # self.output_layer = tf.nn.sigmoid(matmul_fc, name = 'sigmoid')
        # self.cost = tf.square(tf.subtract(self.y_, self.output_layer, name = 'subtract'), name = 'squared_diff')
        ###############################################
        self.output = tf.reshape(matmul_fc, [-1, 8 * 8, CHARS_LENGTH])
        self.cost = tf.losses.sparse_softmax_cross_entropy(self.y_, self.output)

        # all_zeros = tf.zeros(tf.shape(self.y_))
        # zeroed_y_ = tf.where(self.y_ < 0, all_zeros, self.y_)

        # self.mean_cost = tf.reduce_mean(self.cost, axis=[1])
        LEARNING_RATE = 1e-5
        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        # self.acc = tf.reduce_mean(self.cost)

def retrieve_data(quant, train=True):
    file_dict = dict()
    # json_file_name = 'file_dict.json'
    if train:
        json_file_name = './Chess positions/file_dict_train.json'
    else:
        json_file_name = './Chess positions/file_dict_test.json'
    json_file = open(json_file_name, 'r')
    json_data = json_file.read()
    json_file.close()

    file_dict = json.loads(json_data)
    data = []
    for file_name in file_dict:
        fen_string = file_dict[file_name]
        ######################################
        # y_data = make_one_hot(fen_string).ravel()
        ######################################
        y_data = make_out_layer(fen_string).ravel()

        img = mpimg.imread(file_name)
        x_data = img.ravel()
        data.append([x_data, y_data])

    print("{:d} games loaded.".format(len(data)))
    # print(y_data[0], y_data[0].shape)

    return data

def make_out_layer(fen_string):
    out_layer = np.zeros(64)
    for i in range(len(fen_string)):
        char_at_i = fen_string[i]
        out_layer[i] = CHAR_DICT[char_at_i]
    return out_layer

def make_one_hot(fen_string):
    one_hot_array = np.zeros((64, CHARS_LENGTH))
    for i in range(len(fen_string)):
        char_at_i = fen_string[i]
        one_hot_array[i, CHAR_DICT[char_at_i]] = 1
    return one_hot_array

def array_to_fen(out_array):
    s = ""
    for num in out_array:
        s += CHARS[int(num)]
    return s

def train_network(net, saver=None, load_net=True):

    if saver and load_net:
        print("Loading network {}.".format(net.net_file))
        saver.restore(net.sess, net.net_file)
    else:
        init = tf.global_variables_initializer()
        net.sess.run(init)

    QUANTITY = 20000
    NUMBER_OF_ITERATION = 20
    data = retrieve_data(QUANTITY, train=True)
    data = np.array(data)
    np.random.shuffle(data)
    BATCH_SIZE = min(50, len(data))
    # test_partition = len(data) // 10

    # test_data = data[-test_partition:]
    # y_print = test_data[:, 1].tolist()
    # for y in y_print:
    #     print(y)
    # data = data[0 : data.size-test_partition]
    test_data = retrieve_data(QUANTITY, train=False)
    test_data = np.array(test_data)

    # print("Data length:", len(data), len(data[:, 0]), data[:,1][0].shape)
    mean = np.mean(data[:, 0].tolist())
    std = np.std(data[:, 0].tolist())
    # print("Mean:", mean)
    # print("Std:", std)

    for i in range(NUMBER_OF_ITERATION):
        print("Iteration {:d}".format(i+1))

        data_dict = None

        data_length = (len(data) // BATCH_SIZE) * BATCH_SIZE
        # x_data = np.array(data[0][0: BATCH_SIZE])
        # print("x_data shape:")
        # print(x_data.shape)

        for step in range(0, data_length, BATCH_SIZE):
            xy_data = data[step: step + BATCH_SIZE]
            x_data = ((xy_data[:, 0] - mean) / std).tolist()
            y_data = xy_data[:, 1].tolist()
            # print("y data length:", len(y_data))
            # x_data = (np.array(xy_data[:, 0].tolist()) - 180.5) / 65.5
            # y_data = np.array(xy_data[:, 1].tolist())
            # print("Mean:", np.mean(x_data))
            # print("Std:", np.std(x_data))

            data_dict = {net.x: x_data, net.y_: y_data}
            net.sess.run(net.train_step, feed_dict=data_dict)

        # x_data = ((data[:, 0] - mean) / std).tolist()
        # y_data = data[:, 1].tolist()
        x_data = ((test_data[:, 0] - mean) / std).tolist()
        y_data = test_data[:, 1].tolist()
        data_dict = {net.x: x_data, net.y_: y_data}
        train_accuracy, inform = net.sess.run([net.cost, net.inform], feed_dict=data_dict)
        print("Accuracy: {:.4f}".format(train_accuracy))
        # print("Accuracy: {:.4f}".format(train_accuracy[0]))
        if (i+1)%10 == 0:
            # out_array = out_array.reshape(-1, 64)
            ########################################
            # expected = np.argmax(y_data[0].reshape(-1,13), axis=1)
            # print("Expect:", array_to_fen(expected))
            ########################################
            count = 0
            size = 0
            for i, out_array in enumerate(inform):
                size += 1
                if array_to_fen(y_data[i]) == array_to_fen(out_array):
                    count += 1
                else:
                    print("Expect:", array_to_fen(y_data[i]))
                    print("Result:", array_to_fen(out_array))
            print("Sucess: {:d} of {:d}".format(count, size))
            # print("Expect:", array_to_fen(y_data[i]))
            # print("Result:", array_to_fen(out_array))
    if saver:
        save_path = saver.save(net.sess, net.net_file)
        print("Model saved in path: %s" % save_path)

    net.sess.close()


def run_network(train_net=True, load_net=True):
    net = Network()
    saver = tf.train.Saver()

    if train_net:
        print("Training network.")
        train_network(net, saver, load_net)
    else:
        print("Testing accuracy on", net.net_file)
        saver.restore(net.sess, net.net_file)
        # simulate_minesweep_solving(net, 4000)

if __name__ == '__main__':
    start_time = time.time()

    train_net = False
    load_net = False

    train_net = True
    load_net = True

    run_network(train_net=train_net, load_net=load_net)
    process_time = time.time() - start_time
    print("{:.2f} seconds altogether.".format(process_time))
