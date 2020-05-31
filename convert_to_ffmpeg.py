#!/usr/bin/env python

import os
import glob
from matplotlib import pylab
from pylab import *
import librosa
import librosa.display
import numpy as np
import pandas as pd
import subprocess as sp
from audioread.exceptions import NoBackendError

# Set the environment variable AUDIO_DIR before running this script.
# This script will convert each mp3 file to its mel spectrogram
# and store it in the same directory

AUDIO_DIR = os.environ.get('AUDIO_DIR')
SAMPLING_RATE = 2000

def convert_to_ffmpeg(file_name_absolute):
	directory = os.path.dirname(file_name_absolute)
	file_name = os.path.splitext(os.path.basename(file_name_absolute))[0]
	ffmpeg_absolute_path = os.path.join(directory, file_name + ".fmpg")
	if os.path.exists(ffmpeg_absolute_path):
		# Don't bother redoing
		return
	else:
		print(f"Trying {file_name_absolute}")

	command = ['ffmpeg',
			   '-i', file_name_absolute,
			   '-f', 's16le',
			   '-acodec', 'pcm_s16le',
			   '-ac', '1']  # channels: 2 for stereo, 1 for mono
	command.extend(['-ar', str(SAMPLING_RATE)])
	command.append('-')
	# 30s at 44.1 kHz ~= 1.3e6
	with open(ffmpeg_absolute_path, 'w') as outf:
		sp.run(command, stdout=outf, bufsize=10 ** 7, stderr=sp.DEVNULL, check=True)


if __name__ == "__main__":
	try:
		for root, dirs, files in os.walk(AUDIO_DIR):
			for file in files:
				#if os.path.exists(os.path.join(root, file.replace(".mp3", ".png")))
				if file.endswith(".mp3"):
					try:
						convert_to_ffmpeg(os.path.join(root, file))
					except NoBackendError:
						print(f"Error on file {file}")
	except FileNotFoundError as e:
		raise
		#print('path variable AUDIO_DIR is not set')	
		#print("AUDIO_DIR  "+AUDIO_DIR)
