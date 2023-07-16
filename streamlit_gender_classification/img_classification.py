import keras
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from keras.preprocessing.image import load_img
import tensorflow_hub as hub
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os 
import h5py
import streamlit as st

def header_male(url):
	 st.sidebar.write(f'<p style="color:#c203fc;font-size:32px;text_align:center;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def header_female(url):
	 st.sidebar.write(f'<p style="color:#227ae6;font-size:32px;text_align:center;border-radius:2%;">{url}</p>', unsafe_allow_html=True)

def classification_machine(image1):

	#loading the gender classifier model
	model = tf.keras.models.load_model('model2.h5') # replace with your model
	shape = (218, 178, 3)  # input shape
	model = tf.keras.Sequential([hub.KerasLayer(model, input_shape=shape)])
	test_image = image1.resize((178, 218))
	test_image = preprocessing.image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)
	class_names = ['female','male']
	predictions = model.predict(test_image)
	scores = tf.nn.softmax(predictions[0])
	scores = scores.numpy()
	image_class = class_names[np.argmax(scores)]

	if image_class == 'female':
		result = header_female("{} with a {:.2f}% confidence.".format(image_class,100*np.max(scores)))
	else:
		result = header_male("{} with a {:.2f}% confidence.".format(image_class,100*np.max(scores)))


	return result
