import cv, convert
import numpy as np

# Use webcam for input if we can make the algorithm run in real time?
CAM_OR_LENA = 'lena'

cv.NamedWindow('Input')
cv.NamedWindow('Output')

if CAM_OR_LENA == 'lena':
	lena = cv.LoadImage('lena.jpg')
	cv.ShowImage('Input', lena)

	# Do processing here...
	array = convert.cv2array(lena)

	cv.ShowImage('Output', lena)
	
	cv.WaitKey(0)
	
else:
	cam = cv.CreateCameraCapture(0)

	while 1:
		frame = cv.QueryFrame(cam)
		if frame == None:
			'Dropped frame...'
			continue
		
		cv.Flip(frame, None, 1)
		cv.ShowImage('Input', frame)
	
		# Do processing here...
		cv.ShowImage('Output', frame)
	
		# Handle events
		k = cv.WaitKey(5)

		if k == 0x1b or k == 0x51 or k == 0x71: # Esc/Q/q to quit
		    break
