import cv2
import numpy as np
from scipy.spatial import distance

img_lester = cv2.imread("Images/Lester512.png")
img_liverpool = cv2.imread("Images/Liverpool.jpg")
img_liverpool_side = cv2.imread("Images/Liverpool_90_left.jpg")

orb = cv2.ORB_create(nfeatures=500)

keypoints_lester, descriptors_lester = orb.detectAndCompute(img_lester, None)
keypoints_liverpool, descriptors_liverpool = orb.detectAndCompute(img_liverpool, None)
keypoints_liverpool_side, descriptors_liverpool_side = orb.detectAndCompute(img_liverpool_side, None)


# to convert number to bit array 8 -> [0, 0, 0, 0, 1, 0, 0, 0]
def bitfield(n):
    result = [int(digit) for digit in bin(n)[2:]]
    len_result = len(result)
    new_result = []
    if len_result < 8:
        amount_of_zeros = 8 - len_result
        for i in range(amount_of_zeros):
            new_result.append(0)
        for i in range(len(result)):
            new_result.append(result[i])
        return new_result
    return result


# modified hamming distance function to return amount of different numbers at collection
def my_hamming_distance(first_value, second_value):
    return float(len(first_value)) * distance.hamming(first_value, second_value)


def return_array_of_256_bits(point):
    result_array = []
    for byte in range(len(point)):
        bit_point = bitfield(point[byte])
        for x in range(len(bit_point)):
            result_array.append(bit_point[x])
    return result_array

def convert_32_descriptors_to_256_bit(descriptors):
    result_matrix = []
    for keypoint in range(len(descriptors)):
        a = return_array_of_256_bits(descriptors[keypoint])
        result_matrix.append(a)
    return result_matrix


# print(descriptors_liverpool_side[0])
# print(return_array_of_256_bits(descriptors_liverpool_side[0]))
# print(return_array_of_256_bits(descriptors_liverpool_side[1]))
# print(convert_32_descriptors_to_256_bit(descriptors_liverpool_side))

descriptors_liverpool_side_bit_format = convert_32_descriptors_to_256_bit(descriptors_liverpool_side)
descriptors_liverpool_bit_format = convert_32_descriptors_to_256_bit(descriptors_liverpool)
descriptors_lester_bit_format = convert_32_descriptors_to_256_bit(descriptors_lester)

# print(descriptors_liverpool_side_bit_format)

first_class: int = 0
second_class: int = 0

etalon_liverpool_plus_lester_bit_format = descriptors_liverpool_bit_format + descriptors_lester_bit_format


def findMinimalHammingDistanceForDescriptor(descriptor, etalons):
    min_distance: float = 257
    index = 1
    index_of_min_distance = 0
    while index < len(etalons):
        current_distance = my_hamming_distance(descriptor, etalons[index])
        # print(f'{index}) Current distance: {current_distance}')
        if min_distance >= current_distance > 0.0:
            index_of_min_distance = index
            min_distance = current_distance
        index += 1
    # print(f'Index of min: {index_of_min_distance}')
    # print(f'Min distance: {min_distance}')
    # print(f'index: {index_of_min_distance}')
    return index_of_min_distance


# print(findMinimalHammingDistanceForDescriptor(li))


for i in range(0, 500):
    index_from_etalon = \
        findMinimalHammingDistanceForDescriptor(descriptors_liverpool_side_bit_format[i],
                                                etalon_liverpool_plus_lester_bit_format)
    if index_from_etalon < 500:
        first_class += 1
    else:
        second_class += 1

print(f'First class: {first_class}')
print(f'Second class: {second_class}')
