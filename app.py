from flask import Flask , render_template
import cv2 , threading , time

app = Flask(__name__)
global predictionLSTM
global predictionCNN
predictionLSTM = "waiting"
predictionCNN = "waiting"

def camera(name):

	# imports
	import tensorflow as tf
	import keras
	from keras.models import load_model
	import numpy as np
	from skimage import data, color, io, filters, morphology,transform, exposure, feature, util

	# variables
	global predictionLSTM
	global predictionCNN
	# global img
	classes = ['bad', 'good']
	sequence = []
	counter = 0
	textLSTM = "waiting"
	textCNN = "waiting"

	# loading the model
	model = load_model('lstm2.h5')
	cnn = load_model('CNN.h5')
	face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')

	# opening a window preview
	# cv2.namedWindow("lstm")
	vc = cv2.VideoCapture(0)

	# open the first frame
	if vc.isOpened():
		rval, frame = vc.read()
	else:
		rval = False

	# font styles
	font = cv2.FONT_HERSHEY_SIMPLEX
	place = (20,int(frame.shape[0]/1.1))
	place2 = (int(frame.shape[1]/2),int(frame.shape[0]/1.1))
	fontScale = 1
	fontColor = (255,255,255)
	lineType = 2

	# show camera
	while rval:

		# get the gray image of the frame
		gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
		# get the faces locations in the frame
		faces = face_cascade.detectMultiScale(gray , scaleFactor=1.1 , minNeighbors=5)

		# loop for every face
		for (x , y , w , h) in faces:

			# get the face img of the frame
			grayFace = gray[y:y+h , x:x+w]

			# set a rectangle around the face
			rectangleColor = (0,0,0)
			rectangleStroke = 3
			cv2.rectangle(frame , (x,y) , (x+w, y+h) , rectangleColor , rectangleStroke)

			# preprocess the image for prediction
			img = np.reshape( cv2.resize(grayFace,(100,100)) ,[100,100,-1])
			outputCNN = cnn.predict( np.reshape(img , [1,100,100,1]) )
			outputCNN = outputCNN[0].tolist()
			textCNN = classes[outputCNN.index(max(outputCNN))]
			predictionCNN = textCNN


			# for LSTM check if there is an sequence of 21
			if counter == 21:
				seq = np.array(sequence)
				outputLSTM = model.predict( np.reshape(seq,[1,21,100,100,-1]) )
				outputLSTM = outputLSTM[0].tolist()
				textLSTM = classes[outputLSTM.index(max(outputLSTM))]
				predictionLSTM = textLSTM
				sequence = []
				counter = 0
			elif counter < 21:
				sequence.append(img)
				counter += 1

		# put the result on the windows
		cv2.putText(frame, 'LSTM : ' + textLSTM , place , font , fontScale , fontColor , lineType)
		cv2.putText(frame, 'CNN : ' + textCNN , place2 , font , fontScale , fontColor , lineType)

		# show the frames
		cv2.imshow("taste expressions", frame)

		# read the next frame
		rval, frame = vc.read()
		key = cv2.waitKey(20)

		# wait if 'esc' is pushed
		if key == 27:
			break

	cv2.destroyWindow("lstm")

	# while rval:
	# 	gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
	# 	faces = face_cascade.detectMultiScale(gray , scaleFactor=1.1 , minNeighbors=5)

	# 	for (x , y , w , h) in faces:
	# 		# print(x , y , w , h)
	# 		roi_gray = gray[y:y+h , x:x+w]
	# 		img = roi_gray
	# 		roi_color = frame[y:y+h , x:x+w]
	# 		# img_item = 'my_image.png'
	# 		# cv2.imwrite(img_item , roi_color)
	# 		color_rectangle = (255,0,0) # BGR
	# 		stroke_rectangle = 2
	# 		cv2.rectangle(frame , (x,y) , (x+w, y+h) , color_rectangle , stroke_rectangle)

	# 		# print(img.shape)
	# 		img = cv2.resize(img,(100,100))
	# 		# print(img.shape)
	# 		img = np.reshape(img,[100,100,-1])
	# 		print(img.shape)
	# 		# img = cv2.equalizeHist(img)
	# 		# print(img.shape)

	# 		if counter == 21:
	# 			seq = np.array(sequence)
	# 			print(seq.shape)
	# 			output = model.predict( np.reshape(seq,[1,21,100,100,-1]) )
	# 			output = output[0].tolist()
	# 			print(output)
	# 			text = classes[output.index(max(output))]
	# 			prediction = classes[output.index(max(output))]
	# 			sequence = []
	# 			counter = 0
	# 		elif counter < 21:
	# 			sequence.append(img)
	# 			# print("appending frame")
	# 			# print(sequence.shape)
	# 			counter += 1


	# 	if counter == 0:
	# 		cv2.putText(frame, 'result' , place , font , fontScale , fontColor , lineType)
	# 		cv2.imshow("preview", frame)
	# 		rval, frame = vc.read()
	# 		key = cv2.waitKey(20)
	# 		time.sleep(1.5)
	# 	else:
	# 		cv2.putText(frame, 'scanning...' , place , font , fontScale , fontColor , lineType)
	# 		cv2.putText(frame, text , place2 , font , fontScale , fontColor2 , lineType)
	# 		cv2.imshow("preview", frame)
	# 		rval, frame = vc.read()
	# 		key = cv2.waitKey(20)
		


		

	# 	if key == 27: # exit on ESC
	# 		break
		
cam = threading.Thread(target=camera , args=(1,))
cam.start()

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('client.html' , LSTM=predictionLSTM , CNN=predictionCNN)
