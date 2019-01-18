from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model
from SamplePreprocessor import preprocess


class FilePaths:
	"filenames and paths to data"
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/'
	fnInfer = '../data/test.png'
	fnCorpus = '../data/corpus.txt'


def train(model, loader):
	"train NN"
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
	earlyStopping = 5 # stop training after this number of epochs without improvement
	while True:
		epoch += 1
		print('Epoch:', epoch)

		# train
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		# validate
		charErrorRate = validate(model, loader)
		
		# if best validation accuracy so far, save model parameters
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
		else:
			print('Character error rate not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break


def validate(model, loader):
	"validate NN"
	print('Validate NN')
	loader.validationSet()
	numCharErr = 0
	numCharTotal = 0
	numWordOK = 0
	numWordTotal = 0
	while loader.hasNext():
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		batch = loader.getNext()
		recognized = model.inferBatch(batch)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
	
	# print validation result
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate


# def infer(model, fnImg):
# 	"recognize text in image provided by file path"
#
# 	img = prepareImg(cv2.imread('Sentences/p3.png'), 50)
# 	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
# 	print('Segmented into %d words' % len(res))
# 	for (j, w) in enumerate(res):
# 		(wordBox, wordImg) = w
# 		(x, y, w, h) = wordBox
# 		cv2.imshow('current_word', wordImg)  # save word
# 		cv2.waitKey(0)
#
# 		img = preprocess(wordImg, Model.imgSize)
# 		batch = Batch(None, [img] * Model.batchSize) # fill all batch elements with same input image
# 		recognized = model.inferBatch(batch) # recognize text
# 		print('Recognized:', '"' + recognized[0] + '"') # all batch elements hold same result

# def infer(model, fnImg):
# 	"recognize text in image provided by file path"
#
# 	image = (cv2.imread('Forms/p1.png'))
# 	im1 = (cv2.imread('Forms/p1.png'))
# 	image = cv2.resize(image, (1000, 1000))
# 	im1 = cv2.resize(image, (1000, 1000))
# 	lines = line_sep(image)
#
# 	cv2.imshow('marked areas', im1)
#
# 	for line in lines:
# 		word, individual_words = word_seperation(line)
# 		cv2.imshow('words',word)
# 		cv2.waitKey(0)
# 		for w in individual_words:
# 			cv2.imshow('word', w)
# 			cv2.waitKey(0)
# 			w = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
# 			img = preprocess(w, Model.imgSize)
# 			batch = Batch(None, [img] * Model.batchSize)  # fill all batch elements with same input image
# 			recognized = model.inferBatch(batch)  # recognize text
# 			print('Recognized:', '"' + recognized[0] + '"')  # all batch elements hold same result

def infer(model, fnImg):
	"recognize text in image provided by file path"
	#image = cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE)
	cv2.cvtColor(fnImg, cv2.COLOR_BGR2GRAY)
	# image = cv2.resize(image,(500,500))
	img = preprocess(image, Model.imgSize)
	batch = Batch(None, [img] * Model.batchSize)
	recognized = model.inferBatch(batch)
	print('Recognized:', '"' + recognized[0] + '"')
	return recognized[0]

def main():
	"main function"
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", help="train the NN", action="store_true")
	parser.add_argument("--validate", help="validate the NN", action="store_true")
	args = parser.parse_args()



	# train or validate on IAM dataset	
	# if args.train or args.validate:
		# load training data, create TF model
	loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

		# save characters of model for inference mode
	open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

		# save words contained in dataset into file
	open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

		# execute training or validation
		# if args.train:
	model = Model(loader.charList)
	train(model, loader)
	# elif args.validate:
		# 	model = Model(loader.charList, mustRestore=True)
		# 	validate(model, loader)

	# infer text on test image
	# else:
	# 	print(open(FilePaths.fnAccuracy).read())
	# 	model = Model(open(FilePaths.fnCharList).read(), mustRestore=True)
	# 	infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
	main()

