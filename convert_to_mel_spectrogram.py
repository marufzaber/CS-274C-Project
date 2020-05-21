#!/usr/bin/env python

import os
import glob
from matplotlib import pylab
from pylab import *
import librosa
import librosa.display
import numpy as np
import pandas as pd

# Set the environment variable AUDIO_DIR before running this script.
# This script will convert each mp3 file to its mal spectrograph
# and store it in the same directory

AUDIO_DIR = os.environ.get('AUDIO_DIR')

def convert_to_mal_spectrograph(file_name_absolute):
	directory = os.path.dirname(file_name_absolute)
	file_name = os.path.splitext(os.path.basename(file_name_absolute))[0]
	mal_spectrograph_absolute_path = os.path.join(directory, file_name + ".png")

	y, sr = librosa.load(file_name_absolute)
	spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
	spect = librosa.power_to_db(spect, ref=np.max)

	librosa.display.specshow(spect)
	pylab.savefig(mal_spectrograph_absolute_path, bbox_inches=None, pad_inches=0)
	pylab.close()

if __name__ == "__main__":
	try:
		for root, dirs, files in os.walk(AUDIO_DIR):
		    for file in files:
		        if file.endswith(".mp3"):
		             convert_to_mal_spectrograph(os.path.join(root, file))	
	except FileNotFoundError:
		print('path variable AUDIO_DIR is not set')	
		print("AUDIO_DIR  "+AUDIO_DIR)