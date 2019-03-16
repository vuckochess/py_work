import numpy as np
import Corner_model_keras as cmk
import Fen_string_manipulations as fen

PIC_WIDTH = 202
PIC_HEIGHT = 146

Options = fen.Options

def keras_model():
    import keras.models
    import keras.layers as L

    model_params = [
        # {'file': 'weights_piece_cnn.h5', 'lr': 0.0001, 'padding': 'same', 'extended': False, 'epochs': 5},
        # {'file': 'temp_weights_piece_extended_cnn.h5', 'lr': 0.001, 'padding': 'same', 'extended': True, 'epochs': 1},
        {'file': 'weights_piece_extended_cnn.h5', 'lr': 0.0001, 'padding': 'same', 'extended': True, 'epochs': 5},
        {'file': 'weights_piece_broad.h5', 'lr': 0.0001, 'padding': 'same', 'extended': True, 'epochs': 5}
    ]
    model_num = Options.option
    model_p = model_params[model_num]

    model_file = model_p['file']
    learning_rate = model_p['lr']
    padding = model_p['padding']
    extended = model_p['extended']
    epochs = model_p['epochs']

    shape = (PIC_WIDTH, PIC_HEIGHT, 1)

    model = keras.models.Sequential()
    model.add(L.Conv2D(16, 11, strides=4, padding=padding, input_shape=shape))
    model.add(L.BatchNormalization())
    model.add(L.Activation('relu'))
    model.add(L.Conv2D(32, 5, strides=3, padding=padding))
    model.add(L.BatchNormalization())
    model.add(L.Activation('relu'))
    model.add(L.Conv2D(64, 3, strides=2, padding=padding))
    model.add(L.BatchNormalization())
    model.add(L.Activation('relu'))
    model.add(L.Flatten())
    if extended:
        model.add(L.Dense(50))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))
    model.add(L.Dense(13))
    model.add(L.Activation('softmax'))
    adam = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=adam,
        metrics=['accuracy'])

    cmk.load_weights(model, model_file)
    # print(model.summary())
    # print('Num of layers:', len(model.layers))

    return model, model_file, epochs


class PiecePredictor:
    BATCH_SIZE = 100

    def __init__(self):
        self.model, self.model_file, self.epochs = keras_model()


    def load_data(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train = x_train.reshape(-1, PIC_WIDTH, PIC_HEIGHT, 1)/255, y_train
        self.x_test, self.y_test = x_test.reshape(-1, PIC_WIDTH, PIC_HEIGHT, 1)/255, y_test

        import keras.utils
        self.y_train_categorical = keras.utils.to_categorical(self.y_train, num_classes=13)
        self.y_test_categorical = keras.utils.to_categorical(self.y_test, num_classes=13)


    def train_model(self):

        from sklearn.utils import class_weight
        class_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(self.y_train),
            self.y_train.flatten()
            )
        print('Class weights:', class_weights)

        self.model.fit(
            self.x_train,
            self.y_train_categorical,
            class_weight=class_weights,
            batch_size=PiecePredictor.BATCH_SIZE,
            epochs=self.epochs,
            validation_split=0.1,
            shuffle=True,
            verbose=2
        )

        self.model.save_weights(self.model_file)
        print("Saved model weights to disk.")


    def evaluate_model(self):
        test_accuracy = self.model.evaluate(
            self.x_test,
            self.y_test_categorical,
            verbose=0
        )
        print('Test set accuracy: {:.5f} error rate, {:.2f}% accurate prediction'.format(test_accuracy[0], test_accuracy[1]*100))


    def show_predicted_data(self):
        y_pred = self.model.predict(self.x_test)
        y_ = np.argmax(y_pred, axis=1)
        y_expected = self.y_test.flatten()
        accuracy = np.equal(y_expected, y_)
        mistake_num = np.count_nonzero(1 - accuracy)
        print('Number of examples: {:d}, number of mistakes: {:d}'.format(y_.size, mistake_num))
        errored = []
        x_array = self.x_test.reshape(-1, PIC_WIDTH, PIC_HEIGHT)
        for i in range(len(accuracy)):
            if accuracy[i] == False:# or y_pred[i][y_[i]]<0.96:
                errored.append([y_expected[i], y_[i], x_array[i], y_pred[i]])

        return errored
