from numpy import *
from collections import namedtuple
import cv

# set_printoptions(threshold='nan')

def hyst(input, t1, t2):
	if t1 < t2:
		t1, t2 = t2, t1
	abovet2 = input > t2
	abovet1r, abovet1c = where(input > t1)
	
	# Struggling to convert this matlab function to something in python...
	# return bwselect(input, abovet2, abovet1c, abovet1r, 8)
	
def nonmax(input, orient, radius):
	
	# Error checking
	if size(input) != size(orient):
		print 'Error in non-maximal suppression: Image and orientation are different sizes...\n'
		return input
	if radius < 1:
		print 'Error in non-maximal suppression: Radius must be >= 1.'
		return input
		
	print '\nPerforming non-maximal suppression.  This could take a while...\n'
		
	rows,cols = input.shape
	im = zeros((rows,cols))
	
	iradius = int(ceil(radius))
	angle = arange(181)*pi/180
	xoff = radius*cos(angle)
	yoff = radius*sin(angle)
	
	hfrac = xoff - floor(xoff)
	vfrac = yoff - floor(yoff)
	
	for row in range(iradius, rows - iradius):
		for col in range(iradius, cols - iradius):
			Or = orient[row,col]
			x = col + xoff[Or]
			y = row - yoff[Or]
			
			fx = floor(x)
			cx = ceil(x)
			fy = floor(y)
			cy = ceil(y)
			tl = input[fy,fx]
			tr = input[fy,cx]
			bl = input[cy, fx]
			br = input[cy, cx]
			
			upperavg = tl + hfrac[Or]*(tr-tl)
			loweravg = bl + hfrac[Or]*(br-bl)
			v2 = upperavg + vfrac[Or]*(loweravg - upperavg)
			
			if input[row,col] > v2:
				im[row,col] = input[row,col]
				
	# Missing a final step here which removes some repeated local maxima...		
	return im

def phasecong(input, nscale = 4, norient = 6, minWaveLength = 3, 
			  mult = 2.1, sigmaOnf = 0.55, k = 2.0, cutOff = 0.5, 
			  g = 10, noiseMethod = -1):
			
	print '***********************************************'
	print '*    Running phase congruency algorithm...    *'
	print '*    Originally developed by Peter Kovesi.    *'
	print '***********************************************\n'
	
	epsilon = 0.0001
	rows, cols = input.shape
	imagefft = fft.fft2(input)

	zero = zeros((rows,cols))
	EO = zeros((nscale, norient), dtype = object)
	PC = zeros((norient), dtype = object)
	covx2 = zero
	covy2 = zero
	covxy = zero
	
	EnergyV = zeros((3, rows,cols))
	pcSum = zeros((rows,cols))
	
	if cols%2:
		rangex = arange(-(cols-1)/2, (cols-1)/2 + 1 )/float(cols-1)
	else:
		rangex = arange(-cols/2, cols/2)/float(cols)
	
	if rows%2:
		rangey = arange(-(rows-1)/2, (rows-1)/2 + 1)/float(rows-1)
	else:
		rangey = arange(-rows/2, rows/2)/float(rows)

		
	x,y = meshgrid(rangex, rangey)
	radius = sqrt(x**2 + y**2)
	theta = arctan2(-y, x)

	lp = fft.ifftshift(1.0 / (1.0 + (radius / 0.45)**(2*15)))
	
	radius = fft.ifftshift(radius)
	theta = fft.ifftshift(theta)
	radius[0,0] = 1
	sintheta = sin(theta)
	costheta = cos(theta)
	
	logGabor = zeros(nscale, dtype = object)
	
	for s in range(nscale):
		wavelength = minWaveLength*(mult**s)
		fo = 1.0/wavelength
		logGabor[s] = exp((-(log(radius/fo))**2) / (2* log(sigmaOnf)**2))
		logGabor[s] = logGabor[s]*lp
		logGabor[s][0,0] = 0
		
	print 'Initialisation done...\n'
	
	if noiseMethod == -2:
		print 'Don\'t use \'noiseMethod = -2\' yet.  Not 100\% sure it\'s working as it should.\n'

	# Main loop ...
	for o in range(norient):
		print 'Calculating values for orientation %d/%d...' %(o+1, norient)
		
		angl = (o)*pi/norient
		ds = sintheta * cos(angl) - costheta * sin(angl)
		dc = costheta * cos(angl) - sintheta * sin(angl)
		dtheta = abs(arctan2(ds,dc))
		dtheta = clip(dtheta*norient/2, -inf, pi)
		spread = (cos(dtheta) + 1)/2
		
		# cv.NamedWindow('test')
		# print spread
		# exit()
					
		sumE_ThisOrient = zero
		sumO_ThisOrient = zero
		sumAn_ThisOrient = zero
		Energy = zero
		
		for s in range(nscale):
			filter = logGabor[s]*spread
			EO[s,o] = fft.ifft2(imagefft * filter)
			An = abs(EO[s,o])
			
			sumAn_ThisOrient = sumAn_ThisOrient + An
			sumE_ThisOrient = sumE_ThisOrient + real(EO[s,o])
			sumO_ThisOrient = sumO_ThisOrient + imag(EO[s,o])
			if s == 0:
				if noiseMethod == -1:
					tau = median(sumAn_ThisOrient/sqrt(log(4)))
				elif noiseMethod == -2:
					nbins = 50
					mx = amax(sumAn_ThisOrient)
					edges = arange(0, mx, mx/nbins)
					hist = histogram(sumAn_ThisOrient, edges)
					n = bincount(hist[0])
					dum = amax(n)
					ind = argmax(n)
					tau = (edges[ind] + edges[ind+1])/2
				maxAn = An
			else:
				maximum(maxAn, An)
			
		EnergyV[0] = EnergyV[0] + sumE_ThisOrient
		EnergyV[1] = EnergyV[1] + cos(angl)*sumO_ThisOrient
		EnergyV[2] = EnergyV[2] + sin(angl)*sumO_ThisOrient	
		
		XEnergy = sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
		MeanE = sumE_ThisOrient / XEnergy
		MeanO = sumO_ThisOrient / XEnergy
		
		for s in range(nscale):
			E = real(EO[s,o])
			O = imag(EO[s,o])
			Energy = Energy + E*MeanE + O*MeanO - abs(E*MeanO - O*MeanE)
				
		if noiseMethod >= 0:
			T = noiseMethod
		else:
			totalTau = tau*(1 - (1/mult)**nscale)/(1-(1/mult))
			EstNoiseEnergyMean = totalTau*sqrt(pi/2)
			EstNoiseEnergySigma = totalTau*sqrt((4-pi)/2)
			T = EstNoiseEnergyMean + k*EstNoiseEnergySigma
			
		Energy = maximum(Energy - T, 0)
		width = (sumAn_ThisOrient/(maxAn + epsilon) - 1) / (nscale - 1)
		weight = 1.0 / (1 + exp((cutOff-width)*g))
		
		PC[o] = weight*Energy/sumAn_ThisOrient
		pcSum = pcSum+PC[o]
		
		covx = PC[o]*cos(angl)
		covy = PC[o]*sin(angl)
		covx2 = covx2 + covx**2
		covy2 = covy2 + covy**2
		covxy = covxy + covx*covy
			
	covx2 = covx2/(norient/2)
	covy2 = covy2/(norient/2)
	covxy = 4*covxy/norient;
	denom = sqrt(covxy**2 + (covx2 - covy2)**2) + epsilon
	M = (covy2+covx2 + denom)/2
	m = (covy2+covx2 - denom)/2
	
	Or = arctan2(EnergyV[2], EnergyV[1])
	Or[Or<0] = Or[Or<0] + pi
	
	Or = around(Or*180/pi)
	
	OddV = sqrt(EnergyV[1]**2 + EnergyV[2]**2)
	featType = arctan2(EnergyV[0], OddV)
	
	print '\nDone!'
		
	phase_info = namedtuple('Phase', ['M', 'm', 'Or', 'featType', 'PC', 'EO', 'T', 'pcSum'])
	return phase_info(M, m, Or, featType, PC, EO, T, pcSum)
	
if __name__ == '__main__':
	import pickle, cv
	# Load image
	file = open('numpy_image', 'r')
	im = pickle.load(file)
	file.close()
	
	# Test array
	# im = arange(0,16).reshape(4,4)
	
	# im = cv.LoadImageM('/Users/Tom/Desktop/letter.gif', cv.CV_LOAD_IMAGE_UNCHANGED)
	# grey = cv.CreateMat(im.height, im.width, cv.CV_8UC1)
	# cv.CvtColor(im, grey, cv.CV_RGB2GRAY)
	phase_data = phasecong(im)
	
	# Access data like so...
	edges = phase_data.M
	corners = phase_data.m
	
	# edges = nonmax(edges, phase_data.Or, 1.)
	# edges = hyst(edges, 0.15, 0.3)
		
	edges = cv.fromarray(edges)
	corners = cv.fromarray(corners)
	
	# Display result
	cv.NamedWindow('Output')
	cv.ShowImage('Output', edges)
	cv.WaitKey(0)