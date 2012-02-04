import cv
import numpy as np
from convert import *

# Use webcam for input if we can make the algorithm run in real time?
CAM_OR_LENA = 'lena'

def phaseconv(input, nscale = 4, norient = 6, minWaveLength = 3, 
			  mult = 2.1, sigmaOnf = 0.55, k = 2.0, cutOff = 0.5, 
			  g = 10, noiseMethod = -1):

	epsilon = 0.0001
	rows, cols = input.shape
	imagefft = np.fft.fft2(input)

	zero = np.zeros((rows,cols))
	E0 = np.zeros((nscale, norient), dtype = np.object)
	PC = np.zeros((norient,1), dtype = np.object)
	covx2 = zero
	covy2 = zero
	covxy = zero
	
	EnergyV = np.zeros((rows,cols,3))
	pcSum = np.zeros((rows,cols))
	
	if cols%2:
		rangex = np.arange(-(cols-1)/2, (cols-1)/2 + 1 )/float(cols-1)
	else:
		rangex = np.arange(-cols/2, cols/2)/float(cols)
	
	if rows%2:
		rangey = np.arange(-(rows-1)/2, (rows-1)/2 + 1)/float(rows-1)
	else:
		rangey = np.arange(-rows/2, rows/2)/float(rows)

		
	x,y = np.meshgrid(rangex, rangey)
	radius = np.sqrt(x**2 + y**2)
	theta = np.arctan2(-y, x)
	radius = np.fft.ifftshift(radius)
	theta = np.fft.ifftshift(theta)
	sintheta = np.sin(theta)
	costheta = np.cos(theta)
	
	lp = np.fft.ifftshift(1.0 / (1.0 + (radius / 0.45)**(2*15)))
	radius[0,0] = 1
	
	logGabor = np.zeros(nscale, dtype = np.object)
	
	for s in range(nscale):
		wavelength = minWaveLength*(mult**s)
		fo = 1.0/wavelength
		logGabor[s] = np.exp((-(np.log(radius/fo))**2) / (2* np.log(sigmaOnf)**2))
		logGabor[s] = logGabor[s]*lp
		logGabor[s][0,0] = 0
	
	for o in range(norient):
		angl = (o)*np.pi/norient
		ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
		dc = costheta * np.cos(angl) - sintheta * np.cos(angl)
		dtheta = np.abs(np.arctan2(ds,dc))
		dtheta = np.clip(dtheta*norient/2, -np.inf, np.pi)
		spread = (np.cos(dtheta) + 1)/2
		
		sumE_ThisOrient = zero
		sumO_ThisOrient = zero
		sumAn_ThisOrient = zero
		Energy = zero
		
		for s in range(nscale):
			filter = logGabor[s]*spread
			E0[s,o] = np.fft.ifft2(imagefft * filter)
			An = np.abs(E0[s,o])
			
			sumAn_ThisOrient = sumAn_ThisOrient + An
			sumE_ThisOrient = sumE_ThisOrient + np.real(E0[s,o])
			sumO_ThisOrient = sumO_ThisOrient + np.imag(E0[s,o])
			
			if s == 0:
				if noiseMethod == -1:
					tau = np.median(sumAn_ThisOrient)
					print tau
			exit()
		
def processing(input):
	'''DO ALL PROCESSING IN HERE...'''
	
	# Convert to greyscale
	grey = cv.CreateMat(input.width, input.height, cv.CV_8UC1)
	edges = cv.CreateMat(input.width, input.height, cv.CV_8UC1)

	cv.CvtColor(input, grey, cv.CV_RGB2GRAY)
	
	# Simple Canny edge detection
	cv.Canny(grey, edges, 70, 100)
	
	# Convert image to numpy array	
	im = np.asarray(grey)
	phaseconv(im)
	
	output = edges
	return output

if __name__ == '__main__':

	cv.NamedWindow('Input')
	cv.NamedWindow('Output')	

	if CAM_OR_LENA == 'lena':
		print 'Press any key to quit..'
	
		lena = cv.LoadImageM('lena.jpg', cv.CV_LOAD_IMAGE_UNCHANGED)
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
