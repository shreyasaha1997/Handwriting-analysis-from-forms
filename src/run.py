from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import os
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model
from SamplePreprocessor import preprocess



def main():
    "main function"
    # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--form", help="train the NN", action="store_true")
    parser.add_argument("--continuous", help="validate the NN", action="store_true")
    args = parser.parse_args()
    if args.form:
        os.system('python infer_from_forms.py')
    if args.continuous:
        os.system('python infer_from_continuous_handwriting.py')            


if __name__ == '__main__':
	main()

