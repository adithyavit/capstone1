#imports
from keras.models import load_model
from data import DataSet
import numpy as np

#declare and load the base model
model_name = 'lstm'

#load the saved model
saved_model = 'lstm-features.012-2.414.hdf5'
model = load_model(saved_model)

#start the recording from picam

#extract features from the first 16 frames of the video 

frames = self.rescale_list(frames, self.seq_length)
sample = data.get_frames_by_filename(video_name, data_type)

#predict and print
prediction = model.predict(np.expand_dims(sample, axis=0))
print(prediction)
data.print_class_from_prediction(np.squeeze(prediction, axis=0))