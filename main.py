import cv2
import numpy as np
from scipy.spatial import distance
import time
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus


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
# before this method returned value between 0 and 1 (1110 and 1011 returned 0.5)
# now my method returns amount of different values (1110 and 1011 return 2)
def my_hamming_distance(first_value, second_value):
    return float(len(first_value)) * distance.hamming(first_value, second_value)


# to convert 32-byte format descriptor to 256-bit format
def return_array_of_256_bits(point):
    result_array = []
    for byte in range(len(point)):
        bit_point = bitfield(point[byte])
        for x in range(len(bit_point)):
            result_array.append(bit_point[x])
    return result_array


# to convert all 32-byte format descriptors to 256-bit format
def convert_32_descriptors_to_256_bit(descriptors):
    result_matrix = []
    for keypoint in range(len(descriptors)):
        a = return_array_of_256_bits(descriptors[keypoint])
        result_matrix.append(a)
    return result_matrix


# to find minimal hamming distance for descriptor when its comparing with collection of descriptors
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
    return index_of_min_distance


# to find minimal hamming distance for descriptor when its comparing with collection clutters`s centers
def findMinimalCenterByHammingDistance(descriptor, etalons):
    min_distance: float = 257
    index = 0
    index_of_minimal_center = 0
    while index < 5:
        current_distance = my_hamming_distance(descriptor, etalons[index])
        if min_distance >= current_distance > 0.0:
            index_of_minimal_center = index
            min_distance = current_distance
        index += 1
    return index_of_minimal_center


# method to find amount of elements in cluster
def calculate_amount_elements_at_cluster(input_array, start_index, finish_index):
    array = [0, 0, 0, 0, 0]
    for i in range(start_index, finish_index):
        if input_array[i] < 5:
            array[input_array[i]] = \
                array[input_array[i]] + 1
    return array


# method to make cluster`s center values equal 0 or 1
def round_array_of_centers(input_array):
    rounded_array = input_array
    for i in range(len(input_array)):
        for j in range(len(input_array[i])):
            if input_array[i, j] < 0.5:
                rounded_array[i, j] = 0
            else:
                rounded_array[i, j] = 1
    return rounded_array


# method returns number of class (1 or 2, 1 - first picture, 2 - second picture)
def get_in_cluster_class_of_descriptor(descriptor, cluster, indexes):
    index_of_descriptor = findMinimalHammingDistanceForDescriptor(descriptor, cluster)
    a = indexes[index_of_descriptor]
    # print(f'Index: {a}')
    if a < 500:
        return 1
    else:
        return 2


def main():
    img_lester = cv2.imread("Images/Leicter_more_white.jpg")
    img_liverpool = cv2.imread("Images/Liverpool_more_white.jpg")
    img_liverpool_side = cv2.imread("Images/Liverpool_more_white_rotate_30.jpg")

    orb = cv2.ORB_create(nfeatures=500)

    keypoints_lester, descriptors_lester = orb.detectAndCompute(img_lester, None)
    keypoints_liverpool, descriptors_liverpool = orb.detectAndCompute(img_liverpool, None)
    keypoints_liverpool_side, descriptors_liverpool_side = orb.detectAndCompute(img_liverpool_side, None)

    # img = cv2.drawKeypoints(img_liverpool, keypoints_liverpool, None)
    # cv2.imshow("leicester", img)
    # cv2.waitKey(0)

    descriptors_liverpool_side_bit_format = convert_32_descriptors_to_256_bit(descriptors_liverpool_side)
    descriptors_liverpool_bit_format = convert_32_descriptors_to_256_bit(descriptors_liverpool)
    descriptors_lester_bit_format = convert_32_descriptors_to_256_bit(descriptors_lester)

    etalon_liverpool_plus_lester = []

    etalon_liverpool_plus_lester_bit_format = descriptors_liverpool_bit_format + descriptors_lester_bit_format

    for liverpool_value in range(len(descriptors_liverpool)):
        etalon_liverpool_plus_lester.append(descriptors_liverpool[liverpool_value])

    for lester_value in range(len(descriptors_lester)):
        etalon_liverpool_plus_lester.append(descriptors_lester[lester_value])

    # FIRST METHOD START
    # start_time = time.time()
    # first_class: int = 0
    # second_class: int = 0
    # for i in range(0, 500):
    #     index_from_etalon = \
    #         findMinimalHammingDistanceForDescriptor(descriptors_liverpool_side_bit_format[i],
    #                                                 etalon_liverpool_plus_lester_bit_format)
    #     if index_from_etalon < 500:
    #         first_class += 1
    #     else:
    #         second_class += 1
    #
    # print(f'First class: {first_class}')
    # print(f'Second class: {second_class}')
    # print("--- %s seconds ---" % (time.time() - start_time))
    # FIRST METHOD END

    # SECOND METHOD START
    start_time = time.time()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(etalon_liverpool_plus_lester_bit_format)
    array_after_clustering = kmeans.labels_
    centers_of_clusters = kmeans.cluster_centers_

    # create 5 arrays which contain indexes of descriptors. It makes to work with each cluster separately
    first_cluster_indexes = []
    second_cluster_indexes = []
    third_cluster_indexes = []
    fourth_cluster_indexes = []
    fifth_cluster_indexes = []

    for v in range(len(array_after_clustering)):
        if array_after_clustering[v] == 0:
            first_cluster_indexes.append(v)
        if array_after_clustering[v] == 1:
            second_cluster_indexes.append(v)
        if array_after_clustering[v] == 2:
            third_cluster_indexes.append(v)
        if array_after_clustering[v] == 3:
            fourth_cluster_indexes.append(v)
        if array_after_clustering[v] == 4:
            fifth_cluster_indexes.append(v)

    first_cluster_values = []
    second_cluster_values = []
    third_cluster_values = []
    fourth_cluster_values = []
    fifth_cluster_values = []

    for index in range(len(first_cluster_indexes)):
        first_cluster_values.append(etalon_liverpool_plus_lester_bit_format[first_cluster_indexes[index]])
    for index in range(len(second_cluster_indexes)):
        second_cluster_values.append(etalon_liverpool_plus_lester_bit_format[second_cluster_indexes[index]])
    for index in range(len(third_cluster_indexes)):
        third_cluster_values.append(etalon_liverpool_plus_lester_bit_format[third_cluster_indexes[index]])
    for index in range(len(fourth_cluster_indexes)):
        fourth_cluster_values.append(etalon_liverpool_plus_lester_bit_format[fourth_cluster_indexes[index]])
    for index in range(len(fifth_cluster_indexes)):
        fifth_cluster_values.append(etalon_liverpool_plus_lester_bit_format[fifth_cluster_indexes[index]])

    print(f'Array of clusters for each descriptor\n{array_after_clustering}')
    # print(f'\nCenters of clusters\n{centers_of_clusters}')
    print('\nAmount of elements at each cluster\n')

    amount_of_elements_in_cluster = \
        calculate_amount_elements_at_cluster(array_after_clustering, 0, len(array_after_clustering))
    amount_of_elements_in_cluster_first_class = \
        calculate_amount_elements_at_cluster(array_after_clustering, 0, (int(len(array_after_clustering) / 2)))
    amount_of_elements_in_cluster_second_class = \
        calculate_amount_elements_at_cluster(array_after_clustering,
                                             (int(len(array_after_clustering) / 2)), len(array_after_clustering))

    print(f'{amount_of_elements_in_cluster}')
    print(amount_of_elements_in_cluster_first_class)
    print(amount_of_elements_in_cluster_second_class)

    centers_of_clusters_rounded = round_array_of_centers(centers_of_clusters)

    # print(f'\nRounded first cluster`s center:\n{centers_of_clusters_rounded[0]}')

    #############################################################
    array_centers_for_each_descriptor = []
    for i in range(0, 500):
        index_from_etalon = \
            findMinimalCenterByHammingDistance(descriptors_liverpool_side_bit_format[i], centers_of_clusters_rounded)
        array_centers_for_each_descriptor.append(index_from_etalon)

    print(f'\nIndexes of the closest class center for each descriptor: {array_centers_for_each_descriptor}')

    classes_array = [0, 0]
    for i in range(len(descriptors_liverpool_side_bit_format)):
        value = 0
        if array_centers_for_each_descriptor[i] == 0:
            value = get_in_cluster_class_of_descriptor(
                descriptors_liverpool_side_bit_format[i], first_cluster_values, first_cluster_indexes
            )
        if array_centers_for_each_descriptor[i] == 1:
            value = get_in_cluster_class_of_descriptor(
                descriptors_liverpool_side_bit_format[i], second_cluster_values, second_cluster_indexes
            )
        if array_centers_for_each_descriptor[i] == 2:
            value = get_in_cluster_class_of_descriptor(
                descriptors_liverpool_side_bit_format[i], third_cluster_values, third_cluster_indexes
            )
        if array_centers_for_each_descriptor[i] == 3:
            value = get_in_cluster_class_of_descriptor(
                descriptors_liverpool_side_bit_format[i], fourth_cluster_values, fourth_cluster_indexes
            )
        if array_centers_for_each_descriptor[i] == 4:
            value = get_in_cluster_class_of_descriptor(
                descriptors_liverpool_side_bit_format[i], fifth_cluster_values, fifth_cluster_indexes
            )

        if value == 1:
            classes_array[0] = classes_array[0] + 1
        if value == 2:
            classes_array[1] = classes_array[1] + 1

    print("--- %s seconds ---" % (time.time() - start_time))
    print(classes_array)
    # SECOND METHOD END


if __name__ == "__main__":
    main()
