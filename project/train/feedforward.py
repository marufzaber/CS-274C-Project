import os
import numpy as np

from project.fma import utils
import multiprocessing.sharedctypes as sharedctypes
import ctypes
import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Reshape

from .base import AUDIO_DIR, AUDIO_META_DIR, preprocess

def train(num_epochs, batch_size, learning_rate, job_dir):
    train, val, test, labels_onehot, features = preprocess()

    #
    # Keras parameters.
    NB_WORKER = len(os.sched_getaffinity(0))  # number of usables CPUs
    params = {'workers': NB_WORKER, 'max_queue_size': 10}


    loader = utils.FeatureLoader(features)
    print('Dimensionality: {}'.format(loader.shape))

    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, loader, loader_type="tid")



    keras.backend.clear_session()

    model = keras.Sequential(
        [Dense(100, input_shape=loader.shape, activation="relu"),
        Dense(100, activation="relu"),
        Dense(labels_onehot.shape[1], activation="softmax")]
    )

    optimizer = keras.optimizers.SGD(lr=learning_rate)#, momentum=0.9, nesterov=True)
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


    #
