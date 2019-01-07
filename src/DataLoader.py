from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess


class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath


class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts


class DataLoader:
	"loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database" 

	def __init__(self, filePath, batchSize, imgSize, maxTextLen):
		"loader for dataset at given location, preprocess images and text according to parameters"

		assert filePath[-1]=='/'

		self.dataAugmentation = False
		self.currIdx = 0
		self.batchSize = batchSize
		self.imgSize = imgSize
		self.samples = []

		uppercase_characters = '../data/hsf_0/upper/'
		lowercase_characters = '../data/hsf_0/lower/'
		chars = set()
		numbers = '../data/hsf_0/digit/'

		for folders in range(1, 27):
			uc = uppercase_characters + str(folders) + '/'
			lc = lowercase_characters + str(folders) + '/'
			for images in os.listdir(uc):
				image_path = uc + images
				if not os.path.getsize(image_path):
					continue
				chars = chars.union(chr(folders + 64))
				self.samples.append(Sample(chr(folders + 64), image_path))

			for images in os.listdir(lc):
				image_path = lc + images
				if not os.path.getsize(image_path):
					continue
				chars = chars.union(chr(folders + 96))
				self.samples.append(Sample(chr(folders + 96), image_path))

		for folders in range(0, 10):
			npath = numbers + str(folders) + '/'
			for images in os.listdir(npath):
				image_path = npath + images
				if not os.path.getsize(image_path):
					continue
				chars = chars.union(str(folders))
				self.samples.append(Sample(str(folders), image_path))
		print(self.samples[0].gtText)
		print(self.samples[0].filePath)

		random.shuffle(self.samples)
		# split into training and validation set: 95% - 5%
		splitIdx = int(0.95 * len(self.samples))
		self.trainSamples = self.samples[:splitIdx]
		self.validationSamples = self.samples[splitIdx:]

		# put words into lists
		self.trainWords = [x.gtText for x in self.trainSamples]
		self.validationWords = [x.gtText for x in self.validationSamples]

		# number of randomly chosen samples per epoch for training 
		self.numTrainSamplesPerEpoch = 250000
		
		# start with train set
		self.trainSet()

		# list of all chars in dataset
		self.charList = sorted(list(chars))

	def truncateLabel(self, text, maxTextLen):
		# ctc_loss can't compute loss if it cannot find a mapping between text label and input 
		# labels. Repeat letters cost double because of the blank symbol needing to be inserted.
		# If a too-long label is provided, ctc_loss returns an infinite gradient
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text

	def trainSet(self):
		"switch to randomly chosen subset of training set"
		self.dataAugmentation = True
		self.currIdx = 0
		random.shuffle(self.trainSamples)
		self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

	
	def validationSet(self):
		"switch to validation set"
		self.dataAugmentation = False
		self.currIdx = 0
		self.samples = self.validationSamples


	def getIteratorInfo(self):
		"current batch index and overall number of batches"
		return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


	def hasNext(self):
		"iterator"
		return self.currIdx + self.batchSize <= len(self.samples)
		
		
	def getNext(self):
		"iterator"
		batchRange = range(self.currIdx, self.currIdx + self.batchSize)
		gtTexts = [self.samples[i].gtText for i in batchRange]
		imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
		self.currIdx += self.batchSize
		return Batch(gtTexts, imgs)


