import imageio
import os
import numpy as np

from project.fma import utils
import multiprocessing.sharedctypes as sharedctypes
import ctypes
import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Conv1D, Conv2D, Conv3D, AveragePooling1D, MaxPooling1D, MaxPooling2D, MaxPooling3D, Flatten, Reshape, BatchNormalization

from .base import AUDIO_DIR, AUDIO_META_DIR, preprocess


class MelSpectrogramLoader(utils.RawAudioLoader):
    def __init__(self, *args, **kwargs):
        super(MelSpectrogramLoader, self).__init__(*args, **kwargs)
        #self.shape = [480, 640, 4]
        self.shape = [480, 640]

    def _load(self, filepath):
        img3d = imageio.imread(filepath, format="PNG-PIL")
        return np.average(img3d, axis=2).reshape([480, 640])
        return img3d
        # layer0 = img3d[:, :, 0]
        # layer1 = img3d[:, :, 1]
        # layer2 = img3d[:, :, 2]
        # layer3 = img3d[:, :, 3]
        # # img2d = np.empty(self.shape, dtype=layer0.dtype)
        # img2d[0::4, :] = layer0
        # img2d[1::4, :] = layer1
        # img2d[2::4, :] = layer2
        # img2d[3::4, :] = layer3
        # import pdb; pdb.set_trace()
        # return img2d



def train(num_epochs, batch_size, learning_rate, job_dir):
    train, val, test, labels_onehot = preprocess()

    #
    # Keras parameters.
    NB_WORKER = len(os.sched_getaffinity(0))  # number of usables CPUs
    params = {'workers': NB_WORKER, 'max_queue_size': 10}


    #print('Dimensionality: {}'.format(loader.shape))
    #SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, MelSpectrogramLoader(), extension="png")


    keras.backend.clear_session()
    # model = keras.Sequential(
    #     [Dense(100, input_shape=loader.shape, activation="relu"),
    #     Dense(100, activation="relu"),
    #     Dense(labels_onehot.shape[1], activation="softmax")]
    # )
    #
    shape = [480, 640]
    model = keras.Sequential()
    model.add(Conv1D(input_shape=shape, filters=8, kernel_size=5, activation="relu"))
    model.add(AveragePooling1D(pool_size=(2,)))
    model.add(BatchNormalization())

    model.add(Conv1D(filters=8, kernel_size=5, activation="relu"))
    model.add(MaxPooling1D(pool_size=(2,)))
    model.add(BatchNormalization())


    #model.add(Flatten())
    model.add(Conv1D(filters=8, kernel_size=5, activation="relu"))
    model.add(MaxPooling1D(pool_size=(2,)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(labels_onehot.shape[1], activation="softmax"))
    # model = keras.models.Sequential()
    # model.add(Dense(100, input_shape=loader.shape, activation="relu"))
    # #model.add(Activation("relu"))
    # model.add(Dense(100, activation="relu"))
    # # model.add(Activation("relu"))
    # model.add(Dense(labels_onehot.shape[1], activation="softmax"))
    # # model.add(Activation("softmax"))
    # #
    optimizer = keras.optimizers.Adam(lr=learning_rate)#, momentum=0.9, nesterov=True)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #
    model.fit_generator(SampleLoader(train, batch_size=batch_size), train.size/batch_size, epochs=num_epochs, **params)

    #model.save(os.path.join(job_dir, 'model-export'), save_format='tf')
    acc = model.evaluate_generator(SampleLoader(val, batch_size=100), val.size, **params)
    print( "**** VAL ACC ****")
    print(acc)
    #
    #loss = model.evaluate_generator(SampleLoader(test, batch_size=64), test.size, **params)
    #Y = model.predict_generator(SampleLoader(test, batch_size=64), test.size, **params);
    #
    #loss

    #
