# USAGE
# python3 recognition.py --detector face_detection_model --encodings encodings.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import RPi.GPIO as GPIO
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os


# Servo init
servoPIN = 25
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz
p.start(2.5) # Initialization

process_this_frame = True

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()    # usePiCamera=True 
time.sleep(2.0)

# loop over frames from the video file stream
while True:
	# start the FPS throughput estimator
	fps = FPS().start()
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
	
	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()
	rects = []

    # loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

			#append coordinates to rects list
			rects.append(box.astype("int"))

	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
	boxes = [(t, r, b, l) for (l, t, r, b) in rects]

	if process_this_frame:
		# compute the facial embeddings for each face bounding box
		encodings = face_recognition.face_encodings(frame, boxes)
		names = []

		# loop over the facial embeddings
		for encoding in encodings:
			# attempt to match each face in the input image to our known
			# encodings
			matches = face_recognition.compare_faces(data["encodings"],
				encoding, tolerance=0.4)
			name = "Unknown"

			# check to see if we have found a match
			if True in matches:
				# find the indexes of all matched faces then initialize a
				# dictionary to count the total number of times each face
				# was matched
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}

				# loop over the matched indexes and maintain a count for
				# each recognized face face
				for i in matchedIdxs:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1

				# determine the recognized face with the largest number
				# of votes (note: in the event of an unlikely tie Python
				# will select first entry in the dictionary)
				name = max(counts, key=counts.get)
											         				
			# update the list of names
			names.append(name)
	
	process_this_frame = not process_this_frame	
	
	for name in names:
		if name != 'Unknown':		
			#start servo
			#p.ChangeDutyCycle(2.5)
			#time.sleep(5)
			p.ChangeDutyCycle(12.5)
			time.sleep(4)
			print("Hi, "+name)
			p.ChangeDutyCycle(2.5)
			time.sleep(0.5)

	# update the FPS counter
	fps.update()
	
	# stop the FPS counter
	fps.stop()
	
	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		y = top - 10 if top - 10 > 10 else top + 10
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 0, 255), 3)
		cv2.rectangle(frame, (left-2, y-15), (left+80, y+10),
			(0, 0, 255), -1)
		cv2.putText(frame, name, (left+2, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
			
	text = "{:.2f} FPS".format(fps.fps())
	cv2.putText(frame, text, (w-80,20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
#fps.stop()
#print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
