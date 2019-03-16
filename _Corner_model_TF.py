import numpy as np
import tensorflow as tf

PIC_HEIGHT = 384
PIC_WIDTH = 216
CORNERS_DIM = 8

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


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def layer_conv(input_layer, num_in, num_out, padding, stride=4):
    W_conv = weight_variable([padding, padding, num_in, num_out])
    b_conv = bias_variable([num_out])

    convolve = conv2d(input_layer, W_conv, stride) + b_conv
    convolve = tf.layers.batch_normalization(convolve)
    return convolve


def resnet_layer(conv_layer, output_dim):
    temp_layer = layer_conv(
        conv_layer, output_dim, output_dim, padding=3, stride=1)
    temp_layer = tf.nn.relu(temp_layer)
    conv_layer = layer_conv(
        temp_layer, output_dim, output_dim, padding=3, stride=1) + conv_layer
    return tf.nn.relu(conv_layer)


def layer_fully_connected(layer_flatten, dim_in, dim_out):
    W_fc = weight_variable(
        [PIC_HEIGHT * PIC_WIDTH * dim_in, dim_out])
    b_fc = bias_variable([dim_out])

    matmul_fc = tf.matmul(layer_flatten, W_fc) + b_fc
    output_layer = tf.nn.relu(matmul_fc)
    return output_layer


def shuffle_np_data(x_data, y_data):
    print('Start shuffle.')
    permutation = np.random.permutation(len(x_data))
    x_data = x_data[permutation]
    y_data = y_data[permutation]
    print('End shuffle.')
    return x_data, y_data


class Network:
    NETWORK_FILE = "./view.ckpt"

    NUMBER_OF_ITERATION = 50

    def __init__(self):
        self.net_file = Network.NETWORK_FILE
        self.is_training = True
        self.sess = tf.Session()
        self.model()
        self.x_train = []
        self.y_train = []

    def train_data(self, x, y):
        self.x_train = x
        self.y_train = y

    def test_data(self, x, y):
        self.x_test = x
        self.y_test = y

    def model(self):

        self.x = tf.placeholder(
            tf.float32, shape=[None, PIC_WIDTH, PIC_HEIGHT], name="x_data")
        x_image = tf.reshape(
            self.x, [-1, PIC_WIDTH, PIC_HEIGHT, 1])

        self.y_ = tf.placeholder(tf.float32, shape=[None, CORNERS_DIM], name="y_data")

        input_dim = 1
        output_dim = 16
        conv1 = layer_conv(x_image, input_dim, output_dim, padding=11, stride=4)
        conv_layer = tf.nn.relu(conv1)
        #######################################################################################
        # conv_layer = resnet_layer(conv_layer, output_dim)
        #######################################################################################

        input_dim = output_dim
        output_dim *= 2
        conv2 = layer_conv(conv_layer, input_dim,
                           output_dim, padding=5, stride=3)
        conv_layer = tf.nn.relu(conv2)
        #######################################################################################
        # conv_layer = resnet_layer(conv_layer, output_dim)
        #######################################################################################

        input_dim = output_dim
        output_dim *= 2
        conv3 = layer_conv(conv_layer, input_dim,
                           output_dim, padding=3, stride=2)
        conv_layer = tf.nn.relu(conv3)

        flatten = tf.reshape(conv_layer, [-1, 16 * 9 * output_dim])
        flatten = tf.layers.dropout(
            flatten, rate=0.25, training=self.is_training)

        W_fc = weight_variable([16 * 9 * output_dim, CORNERS_DIM], 'W')
        b_fc = bias_variable([CORNERS_DIM], 'b')
        self.output = tf.add(tf.matmul(flatten, W_fc), b_fc, name='matmul')

        error = tf.reduce_mean(tf.square(self.y_ - self.output))
        self.cost = tf.sqrt(error)

        LEARNING_RATE = 1e-5
        self.train_step = tf.train.AdamOptimizer(
            LEARNING_RATE).minimize(self.cost)

    def train_network(self, load_net=False):
        saver = tf.train.Saver()

        if load_net:
            print("Loading network {}.".format(self.net_file))
            saver.restore(self.sess, self.net_file)
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)

        BATCH_SIZE = 100
        Network.NUMBER_OF_ITERATION = 10

        x_train_data = self.x_train
        y_train_data = self.y_train

        mean = np.mean(x_train_data)
        std = np.std(x_train_data)

        for i in range(Network.NUMBER_OF_ITERATION):
            x_train_data, y_train_data = shuffle_np_data(
                x_train_data, y_train_data)
            self.is_training = True
            print("Iteration {:d}".format(i+1))

            for step in range(0, len(x_train_data), BATCH_SIZE):
                x_batch = ((x_train_data[step: step+BATCH_SIZE]-mean)/std)
                y_batch = y_train_data[step: step+BATCH_SIZE]
                y_batch = np.array(y_batch)[:, 0:CORNERS_DIM].tolist()

                data_dict = {self.x: x_batch, self.y_: y_batch}
                step = self.sess.run(self.train_step, feed_dict=data_dict)

            x_test_data = ((self.x_test - mean)/std)
            y_test_data = self.y_test
            y_test_data = np.array(y_test_data)[:, 0:CORNERS_DIM].tolist()
            self.is_training = False

            data_dict = {self.x: x_test_data, self.y_: y_test_data}
            train_accuracy, _ = self.sess.run(
                [self.cost, self.output], feed_dict=data_dict)
            print("Accuracy: {:.4f}".format(train_accuracy))

        if saver:
            save_path = saver.save(self.sess, self.net_file)
            print("Model saved in path: %s" % save_path)

        self.sess.close()

def run_network():
    load_net = False
    load_net = True
    predictor = Network()
    import Corner_network as cn
    x_train, y_train = cn.generate_data(True)
    x_test, y_test = cn.generate_data(False)
    predictor.train_data(x_train, y_train)
    predictor.test_data(x_test, y_test)
    predictor.train_network(load_net=load_net)

