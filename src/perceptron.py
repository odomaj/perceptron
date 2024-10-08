from get_file import get_file
import numpy as np
import sys


POSITIVE = "Y"
NEGATIVE = "N"
DEFAULT_INPUT_FILE_PATH = "../data/train-a1-449.txt"
DEFAULT_OUTPUT_FILE_PATH = "../data/output.txt"


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
    """returns the normalized form of the passed vector"""
    normalized_vector = np.array(vector)
    magnitude = np.sqrt(np.dot(normalized_vector, normalized_vector))
    for i in range(len(normalized_vector)):
        normalized_vector[i] = normalized_vector[i] / magnitude
    return normalized_vector


def normalize_all(vectors: list) -> list:
    """returns an identical list containing every vector
    after being normalized in the passed list"""
    normalized_vectors = []
    for vector in vectors:
        normalized_vectors.append((normalize(vector[0]), vector[1]))
    return normalized_vectors


def get_sign(number: float) -> int:
    """returns the sign of the given number"""
    if number < 0:
        return -1
    return 1


def perceptron(vectors: list) -> np.array:
    """Calculating a vector normal vector to the vector that divides the postive
    and negative labeled vectors"""

    """Algorithm Description
    1. start with a zero vector as the normal vector
    2. check all vectors to find a vector where its dot product with
        the normal vector has a different sign than intended
        2a. if there exists one, add that vector, multiplied by its intended
            sign to the normal vector and restart
    3. if no more vectors exist that do not fit 2, output the normal vector as
        it splits the 2 sets of vectors"""
    # 1
    normal_vector = np.array([0] * 1024)
    i = 0
    while i < len(vectors):
        current_vector = vectors[i]
        # 2
        if current_vector[1] != get_sign(
            np.dot(normal_vector, current_vector[0])
        ):
            # 2a
            normal_vector = np.add(
                normal_vector,
                np.multiply(current_vector[1], current_vector[0]),
            )
            i = 0
        else:
            i += 1
    # 3
    return normalize(normal_vector)


def find_margin(vectors: list, normal_vector: np.array) -> float:
    """Calculating the margin which is the minimum dot
    product between the normal_vector and all vectors"""
    min_dot_product = np.dot(normal_vector, vectors[0][0])
    for vector in vectors:
        dot_product = np.dot(normal_vector, vector[0]) * vector[1]
        min_dot_product = min(min_dot_product, dot_product)
    return min_dot_product


if __name__ == "__main__":
    input_file_path = DEFAULT_INPUT_FILE_PATH
    output_file_path = DEFAULT_OUTPUT_FILE_PATH
    if len(sys.argv) >= 2:
        input_file_path = sys.argv[1]
    elif len(sys.argv) >= 3:
        output_file_path = sys.argv[2]
    vectors = normalize_all(read_file(input_file_path))
    normal_vector = perceptron(vectors)
    margin = find_margin(vectors, normal_vector)
    print(f"The normal vector that cuts the points is: {normal_vector}")
    print(f"The margin of this vector is: {margin}")

    with get_file(output_file_path).open("w") as file:
        for value in normal_vector:
            file.write(f"{value} ")
