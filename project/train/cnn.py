import imageio
import os
import numpy as np

from project.fma import utils
import multiprocessing.sharedctypes as sharedctypes
import ctypes
import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Conv1D, Conv2D, Conv3D, LSTM, RNN, SimpleRNN, AveragePooling1D, MaxPooling1D, MaxPooling2D, MaxPooling3D, Flatten, Reshape,  Permute, BatchNormalization, Dropout
import tensorflow as tf

from .base import AUDIO_DIR, AUDIO_META_DIR, preprocess
from ..tools import graph_generator

class MelSpectrogramLoader(utils.RawAudioLoader):
    def __init__(self, *args, **kwargs):
        super(MelSpectrogramLoader, self).__init__(*args, **kwargs)
        #self.shape = [480, 640, 4]
        #self.shape = [480, 640]
        self.shape = [640, 480, 1]

    def _load(self, filepath):
        img3d = imageio.imread(filepath, format="PNG-PIL")
        #return np.average(img3d, axis=2).reshape([480, 640]).T
        #return img3d
        ret = img3d.transpose([1, 0, 2])
        #return ret
        return ret.mean(axis=2).reshape([640, 480, 1])
        #
        # #ret = img3d.transpose([])
        # layer0 = img3d[:, :, 0]
        # layer1 = img3d[:, :, 1]
        # layer2 = img3d[:, :, 2]
        # layer3 = img3d[:, :, 3]
        # img2d = np.empty(self.shape, dtype=layer0.dtype)
        # img2d[:, 0::4] = layer0
        # img2d[:, 1::4] = layer1
        # img2d[:, 2::4] = layer2
        # img2d[:, 3::4] = layer3
        # return img2d
        # # img2d[0::4, :] = layer0
        # img2d[1::4, :] = layer1
        # img2d[2::4, :] = layer2
        # img2d[3::4, :] = layer3
        # import pdb; pdb.set_trace()
        # return img2d



def train(num_epochs, batch_size, learning_rate, output_dir):
    train, val, test, labels_onehot = preprocess()

    #
    # Keras parameters.
    NB_WORKER = len(os.sched_getaffinity(0))  # number of usables CPUs
    params = {'workers': NB_WORKER, 'max_queue_size': 10}


    #print('Dimensionality: {}'.format(loader.shape))
    #SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, MelSpectrogramLoader(), extension="png")


    class Cell(keras.layers.Layer):
        def __init__(self, units, **kwargs):
            self.units = units
            self.state_size = units
            super(Cell, self).__init__(**kwargs)

        def build(self, input_shape):
            self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                          initializer='uniform',
                                          name='kernel')
            self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                    initializer='uniform',
                                                    name='recurrent_kernel')
            self.built = True

        def call(self, inputs, states):
            prev_output = states[0]
            h = tf.tensordot(inputs, self.kernel, 1)
            output = h + tf.tensordot(prev_output, self.recurrent_kernel, 1)
            return output, [output]

    keras.backend.clear_session()
    shape = [640, 480, 1]
    model = keras.Sequential()

    model.add(Reshape([640, 480]))
    model.add(Conv1D(input_shape=shape, filters=4, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Reshape([638, 4]))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Reshape([319, 4]))

    model.add(Conv1D(filters=4, kernel_size=2, activation="relu"))
    model.add(BatchNormalization())
    model.add(Reshape([318, 4]))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Reshape([159, 4]))

    model.add(Conv1D(filters=4, kernel_size=2, activation="relu"))
    model.add(BatchNormalization())
    model.add(Reshape([158, 4]))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Reshape([79, 4]))

    model.add(Dropout(rate=0.5))
    #model.add(Reshape([]))
    #model.add(SimpleRNN(64))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Reshape([32]))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(labels_onehot.shape[1], activation="softmax"))
    optimizer = keras.optimizers.Adam(lr=learning_rate)#, momentum=0.9, nesterov=True)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    total_epochs = 0


    while (total_epochs < num_epochs):
        training_history = model.fit_generator(SampleLoader(train, batch_size=batch_size), train.size/batch_size, epochs=4, **params).history
        val_loss, val_acc = model.evaluate_generator(SampleLoader(val, batch_size=batch_size), val.size, **params)

        for ep in range(4):
            graph_generator.store_in_csv(total_epochs + ep + 1, batch_size, learning_rate, training_history['accuracy'][ep], training_history['loss'][ep], os.path.join(output_dir, "training_data.csv"))
        graph_generator.store_in_csv(total_epochs + 4, batch_size, learning_rate, val_acc, val_loss, os.path.join(output_dir, "validation_data.csv"))
        total_epochs += 4
    graph_generator.generate_from_csv(os.path.join(output_dir, "training_data.csv"), os.path.join(output_dir, "training_plot.eps"), labels_prefix='Training', color='r')
    graph_generator.generate_from_csv(os.path.join(output_dir, "validation_data.csv"), os.path.join(output_dir, "combined_plot.eps"), labels_prefix='Validation', color='b', epoch_step=4)
    print(model.summary(to_file=os.path.join(output_dir, 'model_summary.txt')))

    #model.save(os.path.join(job_dir, 'model-export'), save_format='tf')
    #loss = model.evaluate_generator(SampleLoader(test, batch_size=64), test.size, **params)
    #Y = model.predict_generator(SampleLoader(test, batch_size=64), test.size, **params);
    #
    #loss

    #
