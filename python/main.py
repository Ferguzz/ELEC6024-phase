#! /usr/bin/python

import cv, sys, numpy, phase, collections, argparse

# Use webcam for input if we can make the algorithm run in real time?
CAM_OR_PIC = 'pic'

def processing(input, args):
	'''DO ALL PROCESSING IN HERE...'''
	
	output = collections.namedtuple('Processing', ['phase_out', 'canny'])
	
	# Convert to greyscale
	grey = cv.CreateMat(input.height, input.width, cv.CV_8UC1)

	cv.CvtColor(input, grey, cv.CV_RGB2GRAY)
	
	# cv.Smooth(grey, smooth, 19)
	# smooth = cv.CreateMat(input.height, input.width, cv.CV_8UC1)
	
	# Simple Canny edge detection
	canny_edges = cv.CreateMat(input.height, input.width, cv.CV_8UC1)
	if args.canny:
		cv.Canny(grey, canny_edges, 70, 100)
	
	# Phase congruency calculation
	# First onvert image to numpy array	
	im = numpy.asarray(grey)
	phase_data = phase.phasecong(im, noiseMethod = args.noise)
	
	if args.corners:
		phase_out = cv.fromarray(phase_data.m)
	else:
		phase_out = cv.fromarray(phase_data.M)
	
	return output(phase_out, canny_edges)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description = 'Show off phase congruency edge detection.')
	parser.add_argument('imagename', nargs = '?', default = '../images/baboon.jpg', help = 'Image file.  Default is baboon.jpg')
	parser.add_argument('--canny', dest = 'canny', action = 'store_true', help = 'Compare with Canny edge detection operator')
	parser.add_argument('--corners', dest = 'corners', action = 'store_true', help = 'Display corners detected by phase congruency operator')
	parser.add_argument('--noise', nargs = '?', metavar = 'int', type = int, dest = 'noise', default = -1, help = 'Phase congruency noise removal method.  -1 = median, - 2 = mode.  Positive values are interpreted as is.')
	args = parser.parse_args()
	
	cv.NamedWindow('Input')
	
	if CAM_OR_PIC == 'pic':
	
		filename = '../images/' + args.imagename
		pic = cv.LoadImageM(filename, cv.CV_LOAD_IMAGE_UNCHANGED)
		cv.ShowImage('Input', pic)
		
 		output = processing(pic, args)

		cv.NamedWindow('Output')	
		
		cv.ShowImage('Output', output.phase_out)
		
		if args.canny:
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
			cv.NamedWindow('Output')	
			cv.ShowImage('Output', output)
	
			# Handle events
			k = cv.WaitKey(5)

			if k == 0x1b: # Esc to quit
			    break