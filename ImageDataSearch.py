"""
Author: Daniel Leef
DATASET: Use emnist-balanced

"""


from __future__ import print_function
import os, sys
from PIL import Image
import scipy.io
import re
import sys
def print_requirements():
    print("This program requires three data files to be present in the current working directory. These files are 'emnist-letters-mapping.txt', 'emnist-letters-test.csv', and 'emnist-letters-train.csv'. These can be downloaded from google docs at these links:\nhttps://drive.google.com/open?id=1jAIcTq-Y3EJjgbt3wkLX7fVi0XmkYwet\nhttps://drive.google.com/open?id=1Cvbbu7bnF6Y5zvdwhx2Hb7xKARVH_M12\nhttps://drive.google.com/open?id=1MfbsIKhw6EE5ybvuGVyTrH6k1AZJI8X7\nIf these three files are not present in the current working directory, the program will throw an error. This right here is not an error message.")
print_requirements()
def get_mapping(mapping_file):
    mapping = {}
    f = open(mapping_file)
    for line in f:
        line =  line.strip().split(" ")
        mapping[int(line[0])] = chr(int(line[1]))
  
    return mapping
def get_data(file_name, iterator, mapping_file):
    if mapping_file == None:
        class NoMapping:
            def __getitem__(self, key):
                return key
        mapping = NoMapping()
    else:
        mapping = get_mapping(mapping_file)
    
    data = []
    f = open(file_name, 'r')
    p = re.compile(',')
    lines = f.readlines()
    for i in iterator:
        l = lines[i]
        #Daniel's code
        example = [int(x) for x in p.split(l.strip())]
        pixels = example[1:]
        letter = example[0]
        size = 25,25
        img = Image.new('L', (28, 28))
        img.putdata(pixels)
        img.thumbnail(size, Image.ANTIALIAS)
        ###
        
        img = img.rotate(-90)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        data.append({"label" : str(mapping[letter]), "image" : img})
    return data
def show(data_array, index):
    data_array[index]["image"].show()
def show2(file_name, index):
    show(get_data(file_name, [index], None), 0)
    
def main():
    data = get_data("emnist-letters-train.csv", range(0,100), "emnist-letters-mapping.txt")
    

 
if __name__ == "__main__":
  main()
