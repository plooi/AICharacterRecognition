"""
Author: Peter Looi

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

"""
def main():
    #you can change num_data if you would like
    #num_data is the number of data points in the training data
    #7000 is a good number because it gives pretty decent accuracy, while not taking too long
    num_training_data = 20000
    
    #you can also change num_validation_data
    #this is the number of data points in the validation data
    #cannot be over 1000, otherwise the 1000th data point onwards will be exactly the same as training data
    num_validation_data = 10
    
    
    #get the training data
    #training data starts from image 1000 
    train = (get_data("emnist-letters-train.csv", range(1000, num_training_data+1000), "emnist-letters-mapping.txt"))
    
    
    
    #get the validation data
    #validation data is the first 50 images (first 1000 reserved)
    validation = (get_data("emnist-letters-train.csv", range(0, num_validation_data), "emnist-letters-mapping.txt"))
    
    #take out whitespace on the sides
    #trim_whitespace(train)
    #trim_whitespace(validation)
    
    
    #convert the PIL images to numpy matricies. Numpy matricies allows for faster calculations
    to_numpy(train)
    to_numpy(validation)
    
    #apply a transformation to the image
    #scale_down with average with final_width 12 seems to help the most
    #final_width = 12
    #scale_down(train, final_width, "average")
    #scale_down(validation, final_width, "average")
    
    
    
    #create a NearestNeighbor object. This object has the functionality for the Nearest Neighbor model
    n = NearestNeighbor()
    
    #incremental_deletion will apply the incremental deletion algorithm on the training dataset to improve performance
    #incremental_deletion(train)
    
    #optimal weight matrix
    #initialize_weighted_matrix(-1, 1.5, 12, 12)
    
    
    #Test the model with the validation data (and feeding in the training data, of course, because this is nearest neighbor)
    acc = n.accuracy(train, validation, 
                k=1, #can change the k value (but I found that k = 1 is usually the best)
                verbose = True, #this tells the algorithm whether or not to print percentage complete
                distance_metric = euclidian_distance)#can change the distance metric
    
    
    #print the final accuracy value
    print("Accuracy = " + str(acc))


"""
Here are the various distance metrics

image1 and image2 are numpy matrixes
"""
def euclidian_distance(image1, image2):
    difs = image1 - image2
    return np.square(difs).sum()
def euclidian_distance_cube(image1, image2):
    difs = image1 - image2
    return np.absolute(np.power(difs, np.full(image1.shape, 3))).sum()
weighted_matrix = None
def initialize_weighted_matrix_sinkholes(width, height, sinkholes : list, scalar = 1, offset = 1.5, print_it=False):
    global weighted_matrix
    weighted_matrix = np.full((height, width),0.0)
    for sinkhole_x, sinkhole_y in sinkholes:
        for x in range(width):
            for y in range(height):
                distance_to_sinkhole_scaled_to_0_to_1 = ( ((y-sinkhole_y)**2 + (x-sinkhole_x)**2)/(width**2 + height**2) )**.3
                weighted_matrix.itemset((y, x), weighted_matrix.item((y,x)) + distance_to_sinkhole_scaled_to_0_to_1 * scalar + offset)
    if print_it: p(weighted_matrix)
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
def initialize_weighted_matrix_random(min, max, width, height):
    global weighted_matrix
    weighted_matrix = np.full((height, width),0.0)
    for x in range(width):
        for y in range(height):
            weight = random()*(max-min) + min
            #print("Weight = " + str(weight))
            weighted_matrix.itemset((y, x), weight)
def weighted_euclidian_distance(image1, image2):
    difs = image1 - image2
    euc_dist = np.square(difs)
    return np.multiply(euc_dist, weighted_matrix).sum()
def l1_distance(image1, image2):
    difs = image1 - image2
    return np.absolute(difs).sum()
class MinkowskiDistance:
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
Instantiate a NearestNeighbor object to access the classifyer
"""
class NearestNeighbor:
    def __init__(self):
        pass
    """
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
    """
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
    
#Performance optimization of 30X - Peter
def to_numpy(data_array):
    for data in data_array:
        data["image"] = np.array(data["image"])
        
    return data_array
def resize(dataset_with_pilimg, final_width, final_height):
    for data in dataset_with_pilimg:
        data["image"] = data["image"].resize((final_width, final_height))

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
        image = image.resize(original_size)
        data["image"] = image
        
        
                    
#Performance optimization by 50% if scaled to 12X12 (maybe an accuracy increase) - Peter
def scale_down(dataset, final_width : "has to be square, sorry, will fix later", method):
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
doesn't help
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
