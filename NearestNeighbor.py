"""
Author: Peter Looi - Wrote all code in this file    
    !Note: Daniel Leef may have his own version of this file, where he added
    modifications to support edited nearest neighbor
    
Defines NearestNeighbor classification functionality 

If this file is run directly, a test will be ran that
determines the overall accuracy of the model. Currently,
the model is set to the optimal hyperparameters. 
Hyperparameters can be modified by modifying the main 
function
"""
import sys
from ImageDataSearch import *
import numpy as np
from time import time
from math import atan
from math import pi
from random import random
from multiprocessing import Pool
from time import sleep
import math


"""
Main function

Tests the nearest neighbor model

If this file is run directly, a test will be ran that
determines the overall accuracy of the model. Currently,
the model is set to the optimal hyperparameters. 
Hyperparameters can be modified by modifying the main 
"""
def main():
    #you can change num_data if you would like
    #num_data is the number of data points in the training data
    #7000 is a good number because it gives pretty decent accuracy, while not taking too long
    num_training_data = 5000
    
    #you can also change num_validation_data
    #this is the number of data points in the validation data
    #cannot be over 1000, otherwise the 1000th data point onwards will be exactly the same as training data
    num_validation_data = 200
    
    
    #get the training data
    #training data starts from image 1000 
    train = (get_data("emnist-letters-train.csv", range(1000, num_training_data+1000), "emnist-letters-mapping.txt"))
    
    
    
    #get the validation data
    #validation data is the first 50 images (first 1000 reserved)
    validation = (get_data("emnist-letters-train.csv", range(0, num_validation_data), "emnist-letters-mapping.txt"))
    
    
    #convert the PIL images to numpy matricies. Numpy matricies allows for faster calculations
    to_numpy(train)
    to_numpy(validation)
    
    
    #apply a transformation to the image
    #scale_down with average with final_width 12 seems to help the most
    #Scale down increases accuracy by about 3%
    final_width = 12
    scale_down(train, final_width, "average")
    scale_down(validation, final_width, "average")
    
    
    
    #create a NearestNeighbor object. This object contains the functionality for the Nearest Neighbor model
    n = NearestNeighbor()
    
    
    #test the model
    acc = n.accuracy(train, validation, 
                k=1, #k=1 is optimal
                verbose = True, distance_metric = MinkowskiDistance(4))#can change the distance metric   
    
    #print accuracy
    print("Accuracy: " + str(acc))
        
    


"""
Here are the various distance metrics

Distance metrics can be passed in (as second order
functions) into a NearestNeighbor
object's "accuracy" or "classify" or "classify_one"
functions under the "distance_metric" optional parameter

image1 and image2 are numpy matrixes
"""

"""
euclidian_distance

Defines the Euclidian distance distance metric 
"""
def euclidian_distance(image1, image2):
    difs = image1 - image2
    return np.square(difs).sum()
"""
euclidian_distance_cube

Defines the Euclidian distance cubed distance
metric Sum((p-q)^3)
This is not an offical/conventional distance 
metric
"""
def euclidian_distance_cube(image1, image2):
    difs = image1 - image2
    return np.absolute(np.power(difs, np.full(image1.shape, 3))).sum()


"""
weighted_matrix

When using weighted euclidian distance, each pixel must
be weighted. This variable, "weighted_matrix", is the
matrix of weights that is referenced
"""
weighted_matrix = None

"""
initialize_weighted_matrix_sinkholes

Initializes the weighted matrix with one or more locations that 
contain low weight values (called "sinkholes", which is not the
official name, I just made that up)

width: int - desired width of matrix
height: int - desired height of matrix
sinkholes: list that contains the x, y value of each sinkhole, represented as tuples
"""
def initialize_weighted_matrix_sinkholes(width, height, sinkholes : list, scalar = 1, offset = 1.5, print_it=False):
    global weighted_matrix
    weighted_matrix = np.full((height, width),0.0)
    for sinkhole_x, sinkhole_y in sinkholes:
        for x in range(width):
            for y in range(height):
                distance_to_sinkhole_scaled_to_0_to_1 = ( ((y-sinkhole_y)**2 + (x-sinkhole_x)**2)/(width**2 + height**2) )**.3
                weighted_matrix.itemset((y, x), weighted_matrix.item((y,x)) + distance_to_sinkhole_scaled_to_0_to_1 * scalar + offset)
    if print_it: p(weighted_matrix)
    
"""
initialize_weighted_matrix

initializes the weighted matrix with either the middle weighted heavier,
or the edges weighted heavier

scalar: float. If it is more positive, the edges will be weighted more. If
    it is more negative, the center will be weighted more
offset: float. The minimum value in the matrix is equal to the offset. 
    Maximum value is the offset + scalar*max_distance_from_center
"""
def initialize_weighted_matrix(scalar, offset, width, height, print_it=False):
    global weighted_matrix
    y_radius = (height-1)/2
    x_radius = (width-1)/2
    weighted_matrix = np.full((height, width),0)
    for x in range(width):
        for y in range(height):
            distance_to_origin_scaled_to_0_to_1 = ((y-y_radius)**2 + (x-x_radius)**2)/(y_radius**2 + x_radius**2)
            weighted_matrix.itemset((y, x), distance_to_origin_scaled_to_0_to_1 * scalar + offset)
    if print_it:print(weighted_matrix)
"""
initialize_weighted_matrix_random

initializes the weighted matrix with random values
min: float - defines the lowest possible value that can be assigned to any cell
max: float - defines the highest possible value that can be assigned to any cell
width: int - desired width of matrix
height: int - desired height of matrix
"""
def initialize_weighted_matrix_random(min, max, width, height):
    global weighted_matrix
    weighted_matrix = np.full((height, width),0.0)
    for x in range(width):
        for y in range(height):
            weight = random()*(max-min) + min
            #print("Weight = " + str(weight))
            weighted_matrix.itemset((y, x), weight)
"""
weighted_euclidian_distance

defines the distance function for weighted euclidian distance,
and relies on the values in the global variable "weighted_matrix"
"""
def weighted_euclidian_distance(image1, image2):
    difs = image1 - image2
    euc_dist = np.square(difs)
    return np.multiply(euc_dist, weighted_matrix).sum()
"""
l1_distance

defines the distance function for taxicab (l1) distance
"""
def l1_distance(image1, image2):
    difs = image1 - image2
    return np.absolute(difs).sum()
"""
MinkowskiDistance

each instance of MinkowskiDistance defines a Minkowski
distance function with a particular "u" value
"""
class MinkowskiDistance:
    """
    init
    
    pass the "u" value (or p value) (whatever you call it)
    into the constructor
    Minkowski distance evaluates to ( Sum((p_i - q_i)^u) )^(1/u)
    """
    def __init__(self, p):
        self.p = float(p)
        self.power_p = None
        self.power_p_is_none = True
    def __call__(self, image1, image2):
        if self.power_p_is_none:
            self.power_p = np.full(image1.shape, self.p)
            self.power_p_is_none = False
            
        difs = image1 - image2
        dist = np.absolute(np.power(difs, self.power_p)).sum() ** (1/self.p)
        return dist
class WeightedMinkowskiDistance:
    def __init__(self, p, weights):
        self.p = float(p)
        self.power_p = None
        self.power_p_is_none = True
        self.weights = weights
    def __call__(self, image1, image2):
        if self.power_p_is_none:
            self.power_p = np.full(image1.shape, self.p)
            self.power_p_is_none = False
            
        difs = image1 - image2
        weighted_difs = np.multiply(difs, self.weights)
        dist = np.absolute(np.power(weighted_difs, self.power_p)).sum() ** (1/self.p)
        return dist

def euclidian_distance_for_angles(image1, image2):
    difs = image1 - image2
    #here we have the raw angle differences
    difs = np.absolute(difs)
    subtract = np.full(image1.shape, 2*pi)
    difs2 = difs - subtract
    difs2 = np.absolute(difs2)
    difs = np.minimum(difs, difs2)
    return np.square(difs).sum()

                            
                    


"""
Instantiate a NearestNeighbor object to access nearest neighbor classifier functionality
"""
class NearestNeighbor:
    def __init__(self):
        pass
    """
    accuracy
    
    train is an array of training data
    test_array is an array of test data
    k = the k value in KNN
    distance_metric specifys how you would like to calculate distance (such as 
    euclidian distance or taxicab distance)
    
    returns a single float value between 0 and 1, representing the accuracy of the 
    nearest neigbor model on the test data
    """
    def accuracy(self, train, test_array, k=1, distance_metric=euclidian_distance, verbose=True):
        results = self.classify(train, test_array, k=k, distance_metric=distance_metric,verbose=verbose)
        total_images = len(test_array)
        total_correct = 0
        for i in range(len(test_array)):
            if get_label(test_array[i]) == results[i]:
                total_correct += 1
        return total_correct/total_images
    def label_wise_accuracy(self, train, test_array, k=1, distance_metric=euclidian_distance, verbose=True):
        results = self.classify(train, test_array, k=k, distance_metric=distance_metric,verbose=verbose)
        totals = {}
        totals_correct = {}
        for i in range(len(test_array)):
            correct_answer = get_label(test_array[i])
            
            if correct_answer in totals: totals[correct_answer] += 1
            else: totals[correct_answer] = 1
            
            if correct_answer == results[i]:
                if correct_answer in totals_correct: totals_correct[correct_answer] += 1
                else: totals_correct[correct_answer] = 1
        total_accuracies = {}
        for label in totals:
            total_accuracies[label] = totals_correct[label]/totals[label]
        
        return total_accuracies
    """
    classify 
    
    train is an array of training data
    test_array is an array of test data
    k = the k value in KNN
    distance_metric specifys how you would like to calculate distance (such as 
    euclidian distance or taxicab distance)
    
    returns an array, where the ith index of the array is classification that 
    the nearest neigbor algorithm has given for the ith test data for
    """
    def classify(self, train, test_array, k=1, distance_metric=euclidian_distance, verbose=True):
        length = len(test_array)
        
        if type(train[0]) == type([]):
            for subtrain in train:
                self.verify_train_test(subtrain, test_array)
        else:
            self.verify_train_test(train, test_array)
        results = []
        
        i = 0
        for test_input in test_array:
            if k == 1:
                result = self.classify_one_k1(train, test_input, distance_metric=distance_metric)#performance optimization by 10%
            else:
                result = self.classify_one(train, test_input, k=k, distance_metric=distance_metric)
            results.append(result)
            if verbose: print(str(i/length* 100) + "%")
            i += 1
        return results
    """
    classify_one
    
    train is an array of training data
    test_array is one test data point
    k = the k value in KNN
    distance_metric specifys how you would like to calculate distance (such as 
    euclidian distance or taxicab distance)
    
    returns the classification that the nearest neigbor algorithm has given for 
    the test data 
    """
    def classify_one(self, train, test, k=1, distance_metric=euclidian_distance):
        top_k = [None]*k
        top_k_distances = [sys.float_info.max]*k
        for train_data in train:
            dist = distance_metric(get_image(train_data), get_image(test))
            
            #see if this training data can be added to the top k list
            for j in range(len(top_k_distances)-1, -1, -1):
                
                
                if dist >= top_k_distances[j]:#distances 0 to j are closer than dist
                    if j+1 < len(top_k_distances):#add it to the top_k_distances list at j+1 (not in 0 to j). If list is not long enough, forget it
                        top_k_distances.insert(j+1, dist)
                        del top_k_distances[-1]
                        
                        top_k.insert(j+1, train_data)
                        del top_k[-1]
                elif j == 0:
                    top_k_distances.insert(j, dist)
                    del top_k_distances[-1]
                    
                    top_k.insert(j, train_data)
                    del top_k[-1]
        
        #now, we will choose the classification that is the majority 
        frequency_dict = {}
        for data in top_k:
            if data == None: continue
            label = get_label(data)
            if label in frequency_dict:
                frequency_dict[label] = frequency_dict[label] + 1
            else:
                frequency_dict[label] = 1
        #find the majority
        classification = None
        frequency = -9999999999999
        for label in frequency_dict:
            freq = frequency_dict[label]
            if freq > frequency:
                frequency = freq
                classification = label
                
        return classification
    def classify_one_k1(self, train, test, distance_metric=euclidian_distance):
        
        top_one = None
        top_distance = sys.float_info.max
        for train_data in train:
            dist = distance_metric(get_image(train_data), get_image(test))
            
            
            if dist < top_distance:
                top_one = train_data
                top_distance = dist
        return top_one["label"]
                        
    def verify_train_test(self, train, test_array):
        height = -1
        width = -1
        for i in range(len(train)):
            data = train[i]
            if width == -1 and height == -1:
                height = get_height(data)
                width = get_width(data)
            elif width == get_width(data) and height == get_height(data):
                pass #good
            else:
                raise Exception("Training data point number " + str(i) + " has dimensions of " + str(get_width(data)) + ", " + str(get_height(data)) + " while the first training data point has dimensions of " + str(width) + ", " + str(height))
        for i in range(len(test_array)):
            data = test_array[i]
            if width == get_width(data) and height == get_height(data):
                pass #good
            else:
                raise Exception("Test data point number " + str(i) + " has dimensions of " + str(get_width(data)) + ", " + str(get_height(data)) + " while the first training data point has dimensions of " + str(width) + ", " + str(height))
  
"""
get_label, get_height, get_width may change, depending if the interface of
the data point changes
"""
def get_label(data):
    return data["label"]
def get_image(data):
    return data["image"]
def get_height(data):
    height, width = data["image"].shape
    return height
def get_width(data):
    height, width = data["image"].shape
    return width
"""
to_numpy

This function takes a dataset of PIL images as input,
and then converts each PIL image into a numpy matrix.
This provides a 30X speed up in performance over just
iterating through each pixel using a double for loop
-Peter
"""
def to_numpy(data_array):
    for data in data_array:
        data["image"] = np.array(data["image"], dtype='float')
        
    return data_array
"""
resize

Input a dataset of PIL images, and this function will
resize each PIL image to the desired width and height
"""
def resize(dataset_with_pilimg, final_width, final_height):
    for data in dataset_with_pilimg:
        data["image"] = data["image"].resize((final_width, final_height))
"""
trim_whitespace

A dataset of PIL images is input into this function
the function will go through each image and find the 
smallest subimage that still contains the entire
letter. The original image will be replaced
"""
def trim_whitespace(dataset : "PIL image dataset"):
    for data in dataset:
        image = data["image"]
        original_size = image.size
        min_x = 99999999999999
        max_x = -9999999999999
        min_y = 99999999999999
        max_y = -9999999999999
        for x in range(original_size[0]):
            for y in range(original_size[1]):
                if image.getpixel((x, y)) > 0:
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y
        x_difference = max_x - min_x
        y_difference = max_y - min_y
        image = image.crop((min_x, min_y, max_x, max_y))
        #print((min_x, min_y, max_x, max_y))
        image = image.resize(original_size)
        data["image"] = image
        
        
"""
scale_down

Input a dataset of numpy matrix images - This function
will scale each numpy matrix down to the desired width.
If scaled down to a 12X12, will provide a 100% performance 
improvement as well as about a 5% improvement in accuracy
"""
def scale_down(dataset, final_width, method):
    smaller_size = final_width
    for data in dataset:
        image = get_image(data)
        #p(image)
        width = get_width(data)
        height = get_height(data)
        block_width = get_width(data)//smaller_size
        #print(block_width)
        m = np.zeros((smaller_size, smaller_size))
        for block_row in range(smaller_size):
            for block_column in range(smaller_size):
                if method == "max":
                    maximum_value = 0
                    for x in range(block_column*block_width, block_column*block_width+block_width):
                        for y in range(block_row*block_width, block_row*block_width+block_width):
                            if image.item((y,x)) > maximum_value:
                                maximum_value = image.item((y,x))
                    m.itemset((block_row, block_column), maximum_value)
                elif method == "average":
                    average_value= 0
                    for x in range(block_column*block_width, block_column*block_width+block_width):
                        for y in range(block_row*block_width, block_row*block_width+block_width):
                            average_value += image.item((y,x))
                    average_value /= block_width**2
                    #print("Avg: ", average_value, "pixel", (image.item((block_row*block_width,block_column*block_width))))
                    m.itemset((block_row, block_column), average_value)
                elif method == "min":
                    min_value = 999999999999999999999
                    for x in range(block_column*block_width, block_column*block_width+block_width):
                        for y in range(block_row*block_width, block_row*block_width+block_width):
                            if image.item((y,x)) < min_value:
                                min_value = image.item((y,x))
                    m.itemset((block_row, block_column), min_value)
                else:
                    raise Exception("method must be 'average' or 'max' or 'min'")
        data["image"] = m
        #p(m)
        #quit()
    return dataset
    
"""
add_noise

takes each pixel in the input dataset (numpy images)
and if the pixel has a value less than 30, the pixel will be
set to a random value between 0 and 30

This function provided no performance improvement
"""
def add_noise(dataset):
    for data in dataset:
        image = get_image(data)
        width = get_width(data)
        height = get_height(data)
        for x in range(width):
            for y in range(height):
                if image.item((y,x)) < 30:
                    image.itemset((y,x), random() * 30)
                
    return dataset
    
"""
filter

Input a dataset of numpy images - Takes each image,
and iterates through the pixels. For each pixel, if it
is greater than a certain threshold, the pixel is set to
255. Otherwise, the pixel is set to 0.
"""
def filter(dataset):
    for data in dataset:
        image = get_image(data)
        width = get_width(data)
        height = get_height(data)


        for x in range(width):
            for y in range(height):
                if image.item((y, x)) > 50:
                    image.itemset((y, x), 255)
                else:
                    image.itemset((y, x), 0)
    return dataset
"""
p

Prints a numpy matrix in a easy-to-read format
"""
def p(matrix, divide_by_pi=False):
    height, width = matrix.shape
    for y in range(height):
        for x in range(width):
            if divide_by_pi:
                chars = len(str(round(matrix.item((y,x))/pi, 3)))
                spaces = 8-chars
                print(str(round(matrix.item((y,x))/pi, 3)) + " "*spaces, end="")
                print("p, ", end="")
            else:
                chars = len(str(round(matrix.item((y,x)), 3)))
                spaces = 5-chars
                print(str(round(matrix.item((y,x)), 3))+ " "*spaces, end="")
                print(", ", end="")
        print()
    print("\n")

"""
image_gradient

Input a numpy matrix dataset - takes each numpy
image and replaces it with a smaller image
that conveys the color gradients of each section]
of the original image
"""
def image_gradient(dataset):
    for data in dataset:
        image = get_image(data)
        width = get_width(data)
        height = get_height(data)
        if width != 25 or height != 25:
            raise Exception()
            
        smaller_width = 8
        block_width = 3
        #p(image)
        m = np.zeros((smaller_width, smaller_width))
        for block_row in range(smaller_width):
            for block_column in range(smaller_width):
                center_x, center_y = block_column*block_width+((block_width-1)/2), block_row*block_width+((block_width-1)/2)
                average_x = 0
                average_y = 0
                total_magnitude_in_block = 0
                for x in range(block_column*block_width, block_column*block_width+block_width):
                    for y in range(block_row*block_width, block_row*block_width+block_width):
                        magnitude = image.item((y,x))
                        average_x += magnitude * x
                        average_y += magnitude * y
                        total_magnitude_in_block += magnitude
                if total_magnitude_in_block == 0:
                    average_x = center_x
                    average_y = center_y
                else:
                    average_x /= total_magnitude_in_block
                    average_y /= total_magnitude_in_block
                
                dy = -(average_y - center_y) #negative because in the image y gets larger as we go down, but in unit circle y gets larger as we go up
                dx = average_x - center_x
                #print("Block", block_row, block_column, "ax=", average_x, "ay=", average_y, "cx", center_x, "cy", center_y)
                #print("Block", block_row, block_column, "dy", dy, "dx", dx)
                if dy == 0 and dx == 0:
                    angle = 0
                elif dx == 0:
                    if dy > 0:
                        angle = pi/2#90deg
                    else:#dy < 0
                        angle = -pi/2#-90deg
                else:
                    angle = atan(dy/dx)
                if dx < 0:
                    angle += pi#+180deg
                
                m.itemset((block_row, block_column), angle)
        #p(m, True)
        #quit()
        data["image"] = m
    return dataset
    


if __name__ == "__main__": main()


"""
def image_gradient(dataset):
    for data in dataset:
        image = get_image(data)
        width = get_width(data)
        height = get_height(data)
        if width != 25 or height != 25:
            raise Exception()
            
        smaller_width = 5
        block_width = 5
        #p(image)
        m = np.zeros((smaller_width, smaller_width))
        for block_row in range(smaller_width):
            for block_column in range(smaller_width):
                center_x, center_y = block_column*block_width+((block_width-1)/2), block_row*block_width+((block_width-1)/2)
                average_x = 0
                average_y = 0
                total_magnitude_in_block = 0
                for x in range(block_column*block_width, block_column*block_width+block_width):
                    for y in range(block_row*block_width, block_row*block_width+block_width):
                        magnitude = image.item((y,x))
                        average_x += magnitude * x
                        average_y += magnitude * y
                        total_magnitude_in_block += magnitude
                if total_magnitude_in_block == 0:
                    average_x = center_x
                    average_y = center_y
                else:
                    average_x /= total_magnitude_in_block
                    average_y /= total_magnitude_in_block
                
                dy = -(average_y - center_y) #negative because in the image y gets larger as we go down, but in unit circle y gets larger as we go up
                dx = average_x - center_x
                #print("Block", block_row, block_column, "ax=", average_x, "ay=", average_y, "cx", center_x, "cy", center_y)
                #print("Block", block_row, block_column, "dy", dy, "dx", dx)
                if dy == 0 and dx == 0:
                    angle = 0
                elif dx == 0:
                    if dy > 0:
                        angle = pi/2#90deg
                    else:#dy < 0
                        angle = -pi/2#-90deg
                else:
                    angle = atan(dy/dx)
                if dx < 0:
                    angle += pi#+180deg
                
                m.itemset((block_row, block_column), angle)
        #p(m, True)
        #quit()
        data["image"] = m
    return dataset


def image_gradient(dataset):
    for data in dataset:
        image = get_image(data)
        width = get_width(data)
        height = get_height(data)
        if width != 25 or height != 25:
            raise Exception()
            
        smaller_width = 5
        block_width = 5
        #p(image)
        m = np.zeros((smaller_width, smaller_width))
        for block_row in range(smaller_width):
            for block_column in range(smaller_width):
                center_x, center_y = block_column*block_width+((block_width-1)/2), block_row*block_width+((block_width-1)/2)
                average_x = 0
                average_y = 0
                total_magnitude_in_block = 0
                for x in range(block_column*block_width, block_column*block_width+block_width):
                    for y in range(block_row*block_width, block_row*block_width+block_width):
                        magnitude = image.item((y,x))
                        average_x += magnitude * x
                        average_y += magnitude * y
                        total_magnitude_in_block += magnitude
                if total_magnitude_in_block == 0:
                    average_x = center_x
                    average_y = center_y
                else:
                    average_x /= total_magnitude_in_block
                    average_y /= total_magnitude_in_block
                
                dy = -(average_y - center_y) #negative because in the image y gets larger as we go down, but in unit circle y gets larger as we go up
                dx = average_x - center_x
                #print("Block", block_row, block_column, "ax=", average_x, "ay=", average_y, "cx", center_x, "cy", center_y)
                #print("Block", block_row, block_column, "dy", dy, "dx", dx)
                if dy == 0 and dx == 0:
                    angle = 0
                elif dx == 0:
                    if dy > 0:
                        angle = pi/2#90deg
                    else:#dy < 0
                        angle = -pi/2#-90deg
                else:
                    angle = atan(dy/dx)
                if dx < 0:
                    angle += pi#+180deg
                
                m.itemset((block_row, block_column), angle)
        #p(m, True)
        #quit()
        data["image"] = m
    return dataset
def main2():
    data = [{"label" : "", "image" : np.matrix([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    
    
    ])}]
    
    image_gradient(data)
    def distance_to_nearest_pixel_in_other(image1, image2, threshold = 50, search_radius=4):
    height, width = image1.shape
    
    
    
    distance = 0
    for x in range(width):
        for y in range(height):
            if image1.item((x,y)) > threshold:
                found_nearby_darkened_pixel = False
                for r in range(search_radius):
                    for compare_x, compare_y in loop(x, y, r):
                        if compare_x >= 0 and compare_x < width and compare_y >= 0 and compare_y < height:
                            if image2.item((compare_x, compare_y)) > threshold:
                                #pixel x,y in image1 is darkened, and there is a nearby
                                #pixel at compare_x, compare_y in image2 that is darkened,
                                #and it is at r radius away
                                distance += ( (compare_y-y)**2 + (compare_x-x)**2 ) ** 4 #HEAVILY punish far away pixels
                                found_nearby_darkened_pixel = True
                                break
                    if found_nearby_darkened_pixel:
                        break
                if not found_nearby_darkened_pixel:
                    distance += 2*(2*(search_radius+1) ** 2)**6#extra penalty for being far
    
    
    temp = image1
    image1 = image2
    image2 = temp
    #now, do the same thing but for image2 on image1
    for x in range(width):
        for y in range(height):
            if image1.item((x,y)) > threshold:
                found_nearby_darkened_pixel = False
                for r in range(search_radius):
                    for compare_x, compare_y in loop(x, y, r):
                        if compare_x >= 0 and compare_x < width and compare_y >= 0 and compare_y < height:
                            if image2.item((compare_x, compare_y)) > threshold:
                                #pixel x,y in image1 is darkened, and there is a nearby
                                #pixel at compare_x, compare_y in image2 that is darkened,
                                #and it is at r radius away
                                distance += ( (compare_y-y)**2 + (compare_x-x)**2 ) ** 4 #HEAVILY punish far away pixels
                                found_nearby_darkened_pixel = True
                                break
                    if found_nearby_darkened_pixel:
                        break
                if not found_nearby_darkened_pixel:
                    distance += 2*(2*(search_radius+1) ** 2)**6#extra penalty for being far
    return distance
def main2():
    image1 = np.matrix([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]])
    image2 = np.matrix([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,1,0,0,0],
    [0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,1,0,0,0,0,0,0,1,0],
    [0,0,1,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]])
    image3 = np.matrix([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]])
    print(distance_to_nearest_pixel_in_other(image1, image2, threshold=0))
def loop(center_x, center_y, radius):
    min_x = center_x - radius
    min_y = center_y - radius
    max_x = center_x + radius
    max_y = center_y + radius
    
    yield min_x, center_y
    yield max_x, center_y
    yield center_x, max_y
    yield center_x, min_y
    
    left_up = 1
    left_down = 1
    right_up = 1
    right_down = 1
    up_left = 1
    up_right = 1
    down_left = 1
    down_right = 1
    
    while True:
        if left_up > radius: break
        yield min_x, center_y+left_up
        left_up += 1
        yield min_x, center_y-left_down
        left_down += 1
        yield max_x, center_y+right_up
        right_up += 1
        yield max_x, center_y-right_down
        right_down += 1
        
        if up_right < radius:
            yield center_x+up_right, max_y
            up_right += 1
            yield center_x-up_left, max_y
            up_left += 1
            yield center_x+down_right, min_y
            down_right += 1
            yield center_x-down_left, min_y
            down_left += 1
        
"""
