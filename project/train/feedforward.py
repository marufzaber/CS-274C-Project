import os
import numpy as np

from project.fma import utils
import multiprocessing.sharedctypes as sharedctypes
import ctypes
import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Reshape

from .base import AUDIO_DIR, AUDIO_META_DIR, preprocess

def train(num_epochs, batch_size, learning_rate, job_dir):
    train, val, test, labels_onehot = preprocess()

    #
    # Keras parameters.
    NB_WORKER = len(os.sched_getaffinity(0))  # number of usables CPUs
    params = {'workers': NB_WORKER, 'max_queue_size': 10}


    loader = utils.FfmpegLoader(sampling_rate=2000)
    print('Dimensionality: {}'.format(loader.shape))
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())



    keras.backend.clear_session()

    model = keras.Sequential(
        [Dense(100, input_shape=loader.shape, activation="relu"),
        Dense(100, activation="relu"),
        Dense(100, activation="relu"),
        Dense(labels_onehot.shape[1], activation="softmax")]
    )
    #
    # model = keras.models.Sequential()
    # model.add(Dense(100, input_shape=loader.shape, activation="relu"))
    # #model.add(Activation("relu"))
    # model.add(Dense(100, activation="relu"))
    # # model.add(Activation("relu"))
    # model.add(Dense(labels_onehot.shape[1], activation="softmax"))
    # # model.add(Activation("softmax"))
    # #
    optimizer = keras.optimizers.SGD(lr=learning_rate)#, momentum=0.9, nesterov=True)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #
    model.fit(SampleLoader(train, batch_size=batch_size), epochs=num_epochs, **params)

    model.save(os.path.join(job_dir, 'model-export'), save_format='tf')
    #loss = model.evaluate_generator(SampleLoader(val, batch_size=64), val.size, **params)
    #loss = model.evaluate_generator(SampleLoader(test, batch_size=64), test.size, **params)
    #Y = model.predict_generator(SampleLoader(test, batch_size=64), test.size, **params);
    #
    # loss

    #
