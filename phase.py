import cv
import numpy as np
from convert import *

# Use webcam for input if we can make the algorithm run in real time?
CAM_OR_LENA = 'lena'

def phaseconv(input, nscale = 4, norient = 6, minWaveLength = 3, 
			  mult = 2.1, sigmaOnf = 0.55, k = 2.0, cutOff = 0.5, 
			  g = 10, noiseMethod = -1):
	
	# Convert image to numpy array
	im = cv2array(input)	
	
	epsilon = 0.0001
	rows, cols = input.width, input.height
	imagefft = np.fft.fft2(im)

	zero = np.zeros((rows,cols))
	E0 = np.zeros((nscale, norient), dtype = np.object)
	PC = np.zeros((norient,1), dtype = np.object)
	covx2 = zero
	covy2 = zero
	covxy = zero
	
	EnergyV = np.zeros((rows,cols,3))
	pcSum = np.zeros((rows,cols))
	
	if cols%2:
		rangex = np.arange(-(cols-1)/2, (cols-1)/2 + 1)/float(cols-1)
	else:
		rangex = np.arange(-cols/2, cols/2 + 1)/float(cols)
	
	if rows%2:
		rangey = np.arange(-(rows-1)/2, (rows-1)/2 + 1)/float(rows-1)
	else:
		rangey = np.arange(-rows/2, rows/2 + 1)/float(rows)
		
	x,y = np.meshgrid(rangex, rangey)
	radius = np.sqrt(x**2 + y**2)
	theta = np.arctan2(-y, x)
	radius = np.fft.ifftshift(radius)
	theta = np.fft.ifftshift(theta)
	sintheta = np.sin(theta)
	costheta = np.cos(theta)
	
	lp = np.fft.ifftshift(1.0 / (1.0 + (radius / 0.45)**(2*15)))
	
	radius[0,0] = 1
	logGabor = np.zeros((1,nscale), dtype = np.object)
	
	for s in range(nscale):
		wavelength = minWaveLength*(mult**s)
		fo = 1.0/wavelength
		logGabor[0,s] = np.exp((-(np.log(radius/fo))**2) / (2* np.log(sigmaOnf)**2))
		logGabor[0,s] = logGabor[0,s]*lp
		logGabor[0,s][0,0] = 0
	
	for 0 in range(norient):
		pass
		
def processing(input):
	'''DO ALL PROCESSING IN HERE...'''
	
	# Convert to greyscale
	grey = cv.CreateImage((input.width, input.height), 8, 1)
	edges = cv.CloneImage(grey)
	cv.CvtColor(input, grey, cv.CV_RGB2GRAY)
	
	# Simple Canny edge detection
	cv.Canny(grey, edges, 70, 100)
	
	phaseconv(grey)
	
	output = edges
	return output

if __name__ == '__main__':

	cv.NamedWindow('Input')
	cv.NamedWindow('Output')	

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
