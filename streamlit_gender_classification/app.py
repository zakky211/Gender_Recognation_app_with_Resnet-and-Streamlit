#importing the libraries
import streamlit as st
import joblib
from PIL import Image
from skimage.transform import resize
import numpy as np
import time
import tensorflow as tf
from img_classification import classification_machine
import cv2

#setting up page title,icon
st.set_page_config(page_title='WhoAmI', page_icon=':woman:', layout='centered', initial_sidebar_state='auto')

#importing my haar classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#title
st.markdown("<h1 style='text-align: center;font-size:50px; color: black;'>Gender Classification App </h1>", unsafe_allow_html=True)

#front image upload and display
image_front = Image.open('test_photos/front.png')
show = st.image(image_front, use_column_width=True)

#short description 
st.sidebar.markdown("<h1 style='text-align: center;font-size:40px; color: black;'>Upload Image </h1>", unsafe_allow_html=True)
st.sidebar.write('Please upload face image (png, jpg, jpeg) you want to predict!')

#disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)

#detect_faces function 
def detect_faces(image):
	new_img = np.array(image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	#faces = face_cascade.detectMultiScale(gray,1.05, 6,(30,30))
	faces  = face_cascade.detectMultiScale(image=gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))
	number_of_faces = len(faces)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:
				 cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 8)

	if number_of_faces > 0:
		cropped_img = img[y:y+h, x:x+w]
	else:
		cropped_img = None 
	return img,faces,number_of_faces,cropped_img 

#uploader
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )
if uploaded_file is not None:
	#that's our uploaded image
	image = Image.open(uploaded_file)

	result_img,result_faces,number_of_faces,cropped_img = detect_faces(image)
	st.image(result_img)
	st.write("Classifying...")
	# st.write('Number of faces found',number_of_faces)

	# if cropped_img is not None:
	# 	st.write("Here's cropped image which should display face...")
	# 	st.image(cropped_img)
	# else:
	# 	st.write("Unfortunately couldn't find face to crop...")

	show.empty()  #that function close image_front photo.
	#st.image(image, caption='Uploaded photo.', use_column_width=True)
	st.write("")

	st.sidebar.write("<h1 style='text-align: center; color: green;'>My prediction is...</h1>", unsafe_allow_html=True)

	label = classification_machine(image) #OKAY SO I WANT TO PASS MY CROPPED IMAGE HERE

with st.sidebar:
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	st.text("")
	# st.sidebar.markdown("<h1 style='text-align: left;font-size:15px; color: black;'>Made by Maciej Gronczynski </h1>", unsafe_allow_html=True)

	"""
	
	"""