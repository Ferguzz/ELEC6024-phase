import cv, sys, numpy, phase

# Use webcam for input if we can make the algorithm run in real time?
CAM_OR_PIC = 'pic'

def processing(input):
	'''DO ALL PROCESSING IN HERE...'''
	
	# Convert to greyscale
	grey = cv.CreateMat(input.height, input.width, cv.CV_8UC1)
	edges = cv.CreateMat(input.height, input.width, cv.CV_8UC1)
	smooth = cv.CreateMat(input.height, input.width, cv.CV_8UC1)

	cv.CvtColor(input, grey, cv.CV_RGB2GRAY)
	
	# Simple Canny edge detection
	cv.Smooth(grey, smooth, 19)
	cv.Canny(grey, edges, 70, 100)
	
	# Convert image to numpy array	
	im = numpy.asarray(grey)
	phase_edges = phase.phasecong(im)

	return cv.fromarray(phase_edges)

if __name__ == '__main__':

	cv.NamedWindow('Input')
	cv.NamedWindow('Output')	

	if CAM_OR_PIC == 'pic':
		print 'Press any key to quit..'
		
		if len(sys.argv) == 1:
			filepath = 'lena.jpg'
		else:
			filepath = sys.argv[1]
	
		pic = cv.LoadImageM(filepath, cv.CV_LOAD_IMAGE_UNCHANGED)
		cv.ShowImage('Input', pic)
 		output = processing(pic)
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