#!/usr/bin/env python

import matplotlib.pyplot as plt

# Set the environment variable AUDIO_DIR before running this script.
# This script will convert each mp3 file to its mel spectrogram
# and store it in the same directory

def generate(history_dict, file):
	
	loss_values = history_dict['loss']
	accuracy_values = history_dict['accuracy']
	epochs = range(1, len(accuracy_values) + 1)
	plt.plot(epochs, loss_values, 'bo', label='loss')
	plt.plot(epochs, accuracy_values, 'b', label='accuracy')
	plt.title('Training loss and accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()
	plt.savefig(file)

