import cv, sys, numpy, phase, collections

# Use webcam for input if we can make the algorithm run in real time?
CAM_OR_PIC = 'pic'

def processing(input, canny = 0):
	'''DO ALL PROCESSING IN HERE...'''
	
	output = collections.namedtuple('Processing', ['phase_data', 'canny'])
	
	# Convert to greyscale
	grey = cv.CreateMat(input.height, input.width, cv.CV_8UC1)

	cv.CvtColor(input, grey, cv.CV_RGB2GRAY)
	
	# cv.Smooth(grey, smooth, 19)
	# smooth = cv.CreateMat(input.height, input.width, cv.CV_8UC1)
	
	# Simple Canny edge detection
	canny_edges = cv.CreateMat(input.height, input.width, cv.CV_8UC1)
	if canny:
		cv.Canny(grey, canny_edges, 70, 100)
	
	# Phase congruency calculation
	# First onvert image to numpy array	
	im = numpy.asarray(grey)
	phase_data = phase.phasecong(im)
	
	return output(phase_data, canny_edges)

if __name__ == '__main__':

	cv.NamedWindow('Input')
	cv.NamedWindow('Output')	

	if CAM_OR_PIC == 'pic':
				
		if len(sys.argv) == 1:
			filepath = 'lena.jpg'
		else:
			filepath = sys.argv[1]
	
		pic = cv.LoadImageM(filepath, cv.CV_LOAD_IMAGE_UNCHANGED)
		cv.ShowImage('Input', pic)
		
		canny = 0
		if len(sys.argv) == 1 or len(sys.argv) == 2:
 			output = processing(pic)
		elif sys.argv[2] == '--with-canny':
			canny = 1
			output = processing(pic, canny = 1)

		phase_edges = cv.fromarray(output.phase_data.M)
		cv.ShowImage('Output', phase_edges)
		
		if canny:
			cv.NamedWindow('Canny')	
			cv.ShowImage('Canny', output.canny)
	
		print '\nPress any key to quit..'
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