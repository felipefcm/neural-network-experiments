
import numpy as np

def loadTraining():

	trainImagesFile = open('./train-images-idx3-ubyte', 'rb')
	trainLabelsFile = open('./train-labels-idx1-ubyte', 'rb')

	trainImages = loadImages(trainImagesFile)
	trainLabels = loadLabels(trainLabelsFile)

	print('Training data loaded')

def loadImages(file):
	
	magic = bytesToInt(file.read(4))
	if(magic != 2051):
		raise RuntimeError('Wrong file for images')

	numImages = bytesToInt(file.read(4))
	numRows = bytesToInt(file.read(4))
	numCols = bytesToInt(file.read(4))

	images = []

	for i in range(numImages):
		
		image = []
		for p in range(numRows * numCols):
			image.append(bytesToInt(file.read(1)))
		
		images.append(image)

	return images

def loadLabels(file):

	magic = bytesToInt(file.read(4))
	if(magic != 2049):
		raise RuntimeError('Wrong file for labels')

	numLabels = bytesToInt(file.read(4))

	labels = []

	for l in range(numLabels):
		labels.append(bytesToInt(file.read(1)))

	return labels

def bytesToInt(bytesObj):
	return int.from_bytes(bytesObj, byteorder='big')
