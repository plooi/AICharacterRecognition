from __future__ import print_function
import os, sys
from PIL import Image
import scipy.io
import re
import sys

def read_train_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  for l in f:
    if (len(data) == 100):
        break
    example = [int(x) for x in p.split(l.strip())]
    letter = example[0]
    pixels = example[1:]
    # Each example is a tuple containing both x (vector) and y (int)
    data.append((letter, pixels))
  return data

def read_validation_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  namehash = {}
  for l in f:
    if (len(data) == 50):
        break
    example = [int(x) for x in p.split(l.strip())]
    letter = example[0]
    pixels = example[1:]
    # Each example is a tuple containing both x (vector) and y (int)
    data.append((letter, pixels))

  return data

def read_test_data(filename):
  f = open(filename, 'r')
  p = re.compile(',')
  data = []
  header = f.readline().strip()
  namehash = {}
  for l in f:
    if (len(data) == 50):
        break
    example = [int(x) for x in p.split(l.strip())]
    letter = example[0]
    pixels = example[1:]
    # Each example is a tuple containing both x (vector) and y (int)
    data.append((letter, pixels))
  return data

def adjustPhotoFormat(train, validation, test):
    size = 25, 25
    outTrain = []
    outValidation = []
    outTest = []

    for (label, pixels) in train:
        img = Image.new('L', (28, 28))
        img.putdata(pixels)
        img.thumbnail(size, Image.ANTIALIAS)
        outTrain.append((label, img))

    for (label, pixels) in validation:
        img = Image.new('L', (28, 28))
        img.putdata(pixels)
        img.thumbnail(size, Image.ANTIALIAS)
        outValidation.append((label, img))
    
    for (label, pixels) in test:
        img = Image.new('L', (28, 28))
        img.putdata(pixels)
        img.thumbnail(size, Image.ANTIALIAS)
        outTest.append((label, img))

    return outTrain, outValidation, outTest

  # Load train and test data.  Learn model.  Report accuracy.
def main():
  train,val,test = get_data(["emnist-balanced-train.csv", "emnist-balanced-test.csv"])
  
  for i in range(0,5):
      img = train[i]
      print((img[0]))
  
  
def get_data(argv):
  # Process command line arguments.
  # (You shouldn't need to change this.)
  outTrain = []
  outValidation = []
  outTest = []
  train = read_train_data(argv[0])
  validation = read_train_data(argv[0])
  test = read_test_data(argv[1])
  outTrain, outValidation, outTest = adjustPhotoFormat(train, validation, test)
  
  return outTrain, outValidation, outTest

if __name__ == "__main__":
  main()
