#!/usr/bin/env python

import matplotlib.pyplot as plt

# Set the environment variable AUDIO_DIR before running this script.
# This script will convert each mp3 file to its mel spectrogram
# and store it in the same directory


def generate(history_dict, file):
	
	loss_values = history_dict['loss']
	val_loss_values = history_dict['accuracy']
	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, loss_values, 'bo', label='Training loss')
	plt.plot(epochs, val_loss_values, 'b', label='Training accuracy')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()
	plt.savefig(file)


