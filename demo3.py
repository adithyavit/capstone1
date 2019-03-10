#imports
from keras.models import load_model
from data import DataSet
import numpy as np
from extractor import Extractor
model = Extractor()
import os
#create npy files using vgg model
# Now loop through and extract features to build the sequence.
folder_name = 'video_frames_2/'
sequence = []
for image in sorted(os.listdir(folder_name)):
    if image ==  '.DS_Store':
        continue
    features = model.extract(folder_name+image)
    sequence.append(features) 
rnn_model = 'data/check_points/lstm-features.032-0.449.hdf5'
rnn_model_load = load_model(rnn_model)
prediction = rnn_model_load.predict(np.expand_dims(sequence, axis=0))
print(prediction)
if(np.argmax(prediction)==1):
    print('fighting video')
else:
    print('normal video')
