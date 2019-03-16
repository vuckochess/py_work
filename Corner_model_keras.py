import numpy as np
import tensorflow as tf
import Fen_string_manipulations as fen

Options = fen.Options

PIC_WIDTH = 384
PIC_HEIGHT = 216
OFFSET = np.array((12, 12))
PADDING = 'SAME'

GRID_SQUARE_SIZE = 24
GRID_HEIGHT_DIM = PIC_HEIGHT//GRID_SQUARE_SIZE
GRID_WIDTH_DIM = PIC_WIDTH//GRID_SQUARE_SIZE
CAT_NUM = 5 # 0 if no corner is present; corner number otherwise

# SCALE_LABELS = [PIC_HEIGHT, PIC_WIDTH]
# SCALE_LABELS = [10, 10]

AVG_CORNER_DATA = [[61, 195, 325, 199, 292, 17, 101, 15],
                    # [PIC_HEIGHT/4, PIC_WIDTH*3/4, PIC_HEIGHT*3/4, PIC_WIDTH*3/4,
                    # PIC_HEIGHT*3/4, PIC_WIDTH/4, PIC_HEIGHT/4, PIC_WIDTH/4],
                    [0, 0, 0, 0, 0, 0, 0, 0]]

def conditional_layer(model, condition, layer1, layer2, params1=(), params2=()):
    if condition:
        model.add(layer1(*params1))
        model.add(layer2(*params2))
    else:
        model.add(layer2(*params2))
        model.add(layer1(*params1))


def load_weights(model, model_file):
    from os import path
    if path.exists(model_file):
        print('Loading model weights from "' + str(model_file) + '" file...')
        model.load_weights(model_file)
        print("Loaded.")
    else:
        print('Could not find file "' + model_file + '".')
    # model.layers[7].set_weights([model.layers[7].get_weights()[0], np.zeros(8)])
    # print('Biases:\r\n', model.layers[7].get_weights()[1])


def scale_data(x_data, y_data):
    x_data = (x_data/255).reshape(-1, PIC_HEIGHT + 2*OFFSET[0], PIC_WIDTH + 2*OFFSET[1], 1)
    from keras.utils.np_utils import to_categorical
    y_data = to_categorical(y_data, num_classes=CAT_NUM)
    return x_data, y_data


def keras_model():
    import keras.models
    import keras.layers as L
    from keras import Model

    model_p = {
        'file': 'weights_model_corners_broad.h5',
        'lr': 0.001,
        'epochs': 5
        }

    model_file = model_p['file']
    learning_rate = model_p['lr']
    epochs = model_p['epochs']

    shape = (PIC_HEIGHT + 2*OFFSET[0], PIC_WIDTH + 2*OFFSET[1], 1)
    activation = L.ReLU()
    inputs = L.Input(shape=shape)

    x_1 = model_branch(inputs, activation, L)
    x_2 = second_branch(inputs, activation, L)

    x = L.concatenate([x_1, x_2], axis=-1)
    x = L.BatchNormalization()(x)
    x = activation(x)

    y = L.Conv2D(CAT_NUM, 1, strides=1, padding='SAME')(x)
    y = L.Activation('softmax')(y)
    # y = L.Activation('sigmoid')(y)
    model = Model(inputs=inputs, outputs=y)
    
    adam = keras.optimizers.Adam(lr=learning_rate)
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=adam)

    # model.summary()
    load_weights(model, model_file)

    return model, model_file, epochs


def model_branch(inputs, activation, L):

    bridge_0 = inputs
    PADDING = 'VALID'
    # from keras import backend as K
    # pad = ((12,12), (12,12))
    # bridge_0 = L.Lambda(lambda inputs: K.spatial_2d_padding(inputs, padding=pad))(bridge_0)

    bridge_0 = L.Conv2D(16, 4, strides=2, padding=PADDING)(bridge_0)
    bridge_0 = L.BatchNormalization()(bridge_0)
    bridge_0 = activation(bridge_0)

    bridge_0 = L.Conv2D(32, 3, strides=2, padding=PADDING)(bridge_0)
    bridge_0 = L.BatchNormalization()(bridge_0)
    bridge_0 = activation(bridge_0)
    # bridge_0 = L.Lambda(lambda inputs: K.spatial_2d_padding(inputs))(bridge_0)
    bridge_0 = L.Conv2D(64, 3, strides=2, padding=PADDING)(bridge_0)
    bridge_0 = L.BatchNormalization()(bridge_0)
    bridge_0 = activation(bridge_0)
    # bridge_0 = L.Lambda(lambda inputs: K.spatial_2d_padding(inputs))(bridge_0)
    bridge_0 = L.Conv2D(64, 5, strides=3, padding=PADDING)(bridge_0)
    # bridge_0 = activation(bridge_0)
    # bridge_0 = L.Conv2D(64, 1, strides=1, padding='SAME')(bridge_0)

    # bridge_1 = L.Conv2D(32, 7, strides=4, padding=PADDING)(inputs)
    # bridge_1 = activation(bridge_1)
    # bridge_1 = L.Conv2D(32, 5, strides=3, padding=PADDING)(bridge_1)
    # bridge_1 = activation(bridge_1)
    # bridge_1 = L.Conv2D(64, 3, strides=2, padding=PADDING)(bridge_1)
    # bridge_1 = activation(bridge_1)
    # bridge_1 = L.Conv2D(32, 1, strides=1, padding='SAME')(bridge_1)
    # x = L.concatenate([bridge_0, bridge_1], axis=-1)
    x = bridge_0
    x = L.BatchNormalization()(x)
    # residue = activation(x)
    # x = L.Conv2D(64, 3, strides=1, padding='SAME')(residue)
    # x = activation(x)
    # x = L.Conv2D(64, 3, strides=1, padding='SAME')(x)
    x = activation(x)
    # x = L.add([x, residue])
    x = L.Conv2D(64, 1, strides=1, padding='SAME')(x)
    return x

def second_branch(inputs, activation, L):
    x = L.Cropping2D(cropping=(OFFSET, OFFSET))(inputs)
    x = L.Conv2D(16, 2, strides=2, padding=PADDING)(x)
    x = L.BatchNormalization()(x)
    x = activation(x)
    x = L.Conv2D(32, 2, strides=2, padding=PADDING)(x)
    x = L.BatchNormalization()(x)
    x = activation(x)
    x = L.Conv2D(64, 2, strides=2, padding=PADDING)(x)
    x = L.BatchNormalization()(x)
    x = activation(x)
    x = L.Conv2D(64, 3, strides=3, padding=PADDING)(x)
    x = L.BatchNormalization()(x)
    x = activation(x)
    x = L.Conv2D(32, 1, strides=1, padding='SAME')(x)
    return x
 

class CornerPredictor:
    BATCH_SIZE = 100
    # Y_SHAPE = ()

    def __init__(self):#, x_train, y_train, x_test, y_test):
        # self.load_data(x_train, y_train, x_test, y_test)
        self.model, self.model_file, self.epochs = keras_model()


    def load_data(self, x_train, y_train, x_test, y_test):
        ############################################################
        # import matplotlib.pyplot as plt
        # plt.imshow(x_train[0][0], cmap='gray')
        # plt.show()

        # rand_bright = 0.7 + np.random.rand(len(x_train))*0.3
        # x_train = (x_train.T*rand_bright).T
        
        # plt.clf()
        # plt.imshow(x_train[0][0], cmap='gray')
        # plt.show()
        ############################################################
        self.x_train, self.y_train = scale_data(x_train, y_train)
        # CornerPredictor.Y_SHAPE = self.y_train.shape
        self.x_test, self.y_test = scale_data(x_test, y_test)


    def train_model(self):
        # from keras.preprocessing.image import ImageDataGenerator
        # rand_bright = 0.8 + np.random.rand(len(self.x_train))*0.2
        # self.x_train = ImageDataGenerator.apply_transform(self.x_train, {'brightness': rand_bright})
        # image_gen = ImageDataGenerator(
        #     brightness_range=[1,1]
        # )
        # image_gen.fit(self.x_train, augment=True)
        
        # _ = self.model.fit_generator(
        #     image_gen.flow(
        #         self.x_train,
        #         self.y_train,
        #         batch_size=CornerPredictor.BATCH_SIZE),
        #     epochs=self.epochs,
        #     shuffle=True,
        #     verbose=2
        # )
        from sklearn.utils import class_weight
        class_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(np.argmax(self.y_train, axis=-1)),
            np.argmax(self.y_train, axis=-1).flatten()
            )
        # class_weights = np.array([1., 1., 1., 1., 1.])
        print('Class weights:', class_weights.round(2))

        _ = self.model.fit(
            self.x_train,
            self.y_train,
            class_weight=class_weights,
            batch_size=CornerPredictor.BATCH_SIZE,
            validation_split=0.1,
            epochs=self.epochs,
            shuffle=True,
            verbose=2
        )

        self.model.save_weights(self.model_file)
        print("Saved model weights to disk.")


    def evaluate_model(self):
        test_error_rate = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=0
        )
        print("Error on the test data: {:.3f}".format(test_error_rate))

    def show_predicted_data(self):
        y_pred = self.model.predict(self.x_test)
        comp_tensors = np.equal(np.argmax(self.y_test, axis=-1), np.argmax(y_pred, axis=-1))
        tensor_equal = np.all(comp_tensors.reshape(-1, GRID_HEIGHT_DIM*GRID_WIDTH_DIM), axis=1)
        compare_arrays = np.average(tensor_equal.astype(int))
        print('Expected:\r\n{}'.format(np.argmax(self.y_test, axis=-1)[0].reshape(GRID_HEIGHT_DIM, GRID_WIDTH_DIM)))
        print('Predicted:\r\n{}'.format(np.argmax(y_pred, axis=-1)[0].reshape(GRID_HEIGHT_DIM, GRID_WIDTH_DIM)))
        print('Predicted wide:\r\n{}'.format(y_pred[0].round(2)))
        # compare_arrays = np.square((self.y_test.reshape(-1, 2) * SCALE_LABELS) -
        #     y_pred.reshape(-1, 2) * SCALE_LABELS).reshape(-1,8)
        print('Accuracy:', compare_arrays.round(2))


# def custom_loss(y_true, y_pred):
#     filter = np.zeros((100,8))
#     print(filter.shape)
#     filter[:, 2] = 1
#     y_pred = y_pred*filter + y_true*(1-filter)

#     import keras.losses as l
#     return l.mse(y_true, y_pred)
