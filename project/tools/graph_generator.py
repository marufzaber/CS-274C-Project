#!/usr/bin/env python

import csv
import matplotlib.pyplot as plt
from os import path

def generate(history_dict, file, labels_prefix='Training', color='b', epoch_step=1):
	
	loss_values = history_dict['loss']
	accuracy_values = [round(float(val), 2) for val in history_dict['accuracy']]
	epochs = [ep*epoch_step for ep in range(1, len(accuracy_values) + 1)]
	#plt.plot(epochs, loss_values, f'{color}o', label=f'{labels_prefix} loss')
	plt.plot(epochs, accuracy_values, color, label=f'{labels_prefix} accuracy')
	plt.title('Loss and accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.ylim((0.00, 1.00))
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

def generate_from_csv(infile, outfile, epoch_step=1, labels_prefix='Training', color='b'):
	with open(infile, "r") as f:
		freader = csv.reader(f, delimiter=',')
		history = {'loss': [],
				   'accuracy': []}
		is_first_line = True
		for row in freader:
			if is_first_line:
				is_first_line = False
				continue
			epoch, batch_size, learning_rate, accuracy, loss = row
			history['accuracy'].append(accuracy)
			history['loss'].append(loss)
	generate(history, outfile, epoch_step=epoch_step, labels_prefix=labels_prefix, color=color)