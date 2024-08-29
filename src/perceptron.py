from get_file import get_file
import numpy as np
import sys

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
    vectors = normalize_all(vectors)


if __name__ == "__main__":
    relative_file_path = DEFAULT_FILE_PATH
    if len(sys.argv) >= 2:
        relative_file_path = sys.argv[1]
    print(perceptron(read_file(relative_file_path)))
