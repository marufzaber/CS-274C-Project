#!/usr/bin/env python

import matplotlib.pyplot as plt
from os import path

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

def store_in_csv(epoch, batch_size, learning_rate, accuracy, loss, file):	
	if path.exists(file) == False:
		with open(file,'w') as f:
			line = "epoch , batch_size, learning_rate, accuracy, loss"
			f.write(line)
			f.write('\n')

	with open(file,'a') as f:
		line = str(epoch) + " , " + str(batch_size) + " , " + str(learning_rate) + " , " + str(accuracy) + " , " + str(loss)
		f.write(line)
		f.write('\n')
