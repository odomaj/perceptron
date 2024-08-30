from get_file import get_file
import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize) #uncomment to show entire np array

POSITIVE = "Y"
NEGATIVE = "N"
DEFAULT_FILE_PATH = "../data/train-a1-449.txt"


def get_array(strings: list) -> np.array:
    """takes a list of strings and outputs an np.array of floats"""
    floats = []
    for string in strings:
        floats.append(float(string))
    return np.array(floats)


def get_value(element: str) -> int:
    """if the passed string is POSITIVE outputs 1
    if the passed string is NEGATIVE outputs -1
    otherwise outputs 0"""
    if element == POSITIVE:
        return 1
    elif element == NEGATIVE:
        return -1
    return 0


def get_value_index(elements: list) -> int:
    """returns the value of the first element from the end
    that is either a POSITIVE or NEGATIVE marker"""
    i = len(elements) - 1
    while i >= 0:
        if get_value(elements[i]) != 0:
            return i
        i -= 1
    return 0


def interpret_line(line: str) -> tuple:
    """takes a line from the input file and outputs a tuple
    containing an np.array of the floats in the line, and the value
    of either the POSITIVE or NEGATIVE flag"""
    contents = line.split(" ")
    i = 0
    for c in contents:
        if get_value(c) != 0:
            i += 1
    value_index = get_value_index(contents)
    return (
        get_array(contents[:value_index]),
        get_value(contents[value_index]),
    )


def read_file(relative_path: str) -> tuple:
    """reads the file at the path relative to the running file and outputs
    the"""
    with get_file(relative_path).open("r") as file:
        lines = file.readlines()
    vectors = []
    for line in lines:
        vectors.append(interpret_line(line))
    return vectors


def normalize(vector: np.array) -> np.array:
    """normalizes the passed vector"""
    magnitude = np.sqrt(np.dot(vector, vector))
    for i in range(len(vector)):
        vector[i] = vector[i] / magnitude


def normalize_all(vectors: list) -> list:
    """normalizes every vector in the passed list"""
    for vector in vectors:
        normalize(vector[0])
    return vectors


def perceptron(vectors: list) -> np.array:
    """Calculating a vector w which is perpendicular to the line that divides the postive and negative labeled vectors"""
    vectors = normalize_all(vectors)
    w = [0] * 1024 #setting vector w to zero
    w = np.array(w)

    v = 0
    
    while v < len(vectors):
        """checking all the signs of the vectors until we hit a case where the signs are all correct
        the loop will exit once all the signs on the vectors match the sign <w,current vector> """
        
        vector = vectors[v] #getting the tuple from the list at the current index

        #calculates the sign from the dot product of the current w and the current vector in the list
        sign = np.dot(w,vector[0]) 
        if(sign >= 0):
            sign = 1
        else:
            sign = -1
        
        #if the signs are not equal the w vector must be corrected
        if(vector[1] != sign):
            vector = (np.multiply(vector[1],vector[0]), vector[1]) #multiplying the vector by its scalar sign
            w = np.add(w,vector[0]) #setting new w to the sum of old w and the current vector
            v = 0 #going back to the beginning of the list to check all the vectors again
        else:
            v += 1
    
    margin = find_margin(vectors,w)
    print("The margin is:")
    print(margin)
    return w

def find_margin(vectors: list, w: np.array):
    """Calculating the margin which is the minimum dot product between w and a vector"""
    products = []

    for vector in vectors:
        """finding the dot product of w and each vector
        then adding the absoulte value of the dot product to the list of dot products"""
        product = np.dot(w,vector[0])
        products.append(abs(product))

    return min(products) #returning the minimum dot product

if __name__ == "__main__":
    relative_file_path = DEFAULT_FILE_PATH
    if len(sys.argv) >= 2:
        relative_file_path = sys.argv[1]
    w = perceptron(read_file(relative_file_path))
    print("The w vector is:")
    print(w)
