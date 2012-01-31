import cv
import numpy as np
from convert import *

# Use webcam for input if we can make the algorithm run in real time?
CAM_OR_LENA = 'cam'

cv.NamedWindow('Input')
cv.NamedWindow('Output')

def processing(input):
	'''DO ALL PROCESSING IN HERE...'''
	
	# Convert to greyscale
	grey = cv.CreateImage((input.width, input.height), 8, 1)
	edges = cv.CloneImage(grey)
	cv.CvtColor(input, grey, cv.CV_RGB2GRAY)
	
	# Simple Canny edge detection
	cv.Canny(grey, edges, 70, 100)
	 
	# Convert to numpy array
	array = cv2array(grey)
	
	output = edges
	return output
	

if CAM_OR_LENA == 'lena':
	print 'Press any key to quit..'
	
	lena = cv.LoadImage('lena.jpg')
	cv.ShowImage('Input', lena)
	
	output = processing(lena)
	cv.ShowImage('Output', output)
	
	cv.WaitKey(0)
	
else:
	print 'Press Esc to quit...'
	
	cam = cv.CreateCameraCapture(0)

	while 1:
		frame = cv.QueryFrame(cam)
		if frame == None:
			'Dropped frame...'
			continue
		
		cv.Flip(frame, None, 1)
		cv.ShowImage('Input', frame)
	
		output = processing(frame)
		cv.ShowImage('Output', output)
	
		# Handle events
		k = cv.WaitKey(5)

		if k == 0x1b: # Esc to quit
		    break
