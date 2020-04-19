from math import sqrt
import numpy as np
import cv2


# geometric features for binary
def roundness(binary_array):
    border = cv2.Canny(binary_array.copy(), 80, 120)
    return (np.count_nonzero(binary_array) * 4 * np.pi) / (np.count_nonzero(border) ** 2)


def extract_geometric_features(binary_image):
    # opencv Hu. We'll use custom, but helps for comparison
    # moments = cv2.moments(binary_image)
    # huMoments = cv2.HuMoments(moments)

    hu_moments = hu(binary_image)
    area = [np.count_nonzero(binary_image)]
    r = [roundness(binary_image)]
    features = area + r + hu_moments
    return features


# KNN
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


def predict(entry, data, selected_features):
    num_neighbors = 1  # Change here the neighbors count for KNN. 1 Works the best in this case.
    distances = list()
    for index, row in data.iterrows():  # Bamboozled
        euclidian = np.linalg.norm(np.array(row[selected_features]) - np.array(entry))
        distances.append([row['tag'], euclidian])

    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    prediction = max(set(neighbors), key=neighbors.count)
    return prediction


# Centralized Hu Moments, invariant to scale, size, position.

def mu(binary_img, p, q):
    size = binary_img.shape
    img_height = size[0]
    img_w = size[1]
    m00 = m_pq(binary_img, 0, 0)
    X = np.tile(np.arange(1, img_w + 1, dtype=np.float64), (img_height, 1))
    Y = np.tile(np.arange(1, img_height + 1, dtype=np.float64), (img_w, 1)).T
    X_cent = [(X - (m_pq(binary_img, 1, 0) / m00)) ** x for x in range(5)]
    Y_cent = [(Y - (m_pq(binary_img, 0, 1) / m00)) ** x for x in range(5)]

    return ((X_cent[p]) * (Y_cent[q]) * binary_img).sum()


def m_pq(binary, a, b):
    size = binary.shape
    img_height = size[0]
    img_w = size[1]
    x = np.tile(np.arange(1, img_w + 1, dtype=np.float64), (img_height, 1))
    y = np.tile(np.arange(1, img_height + 1, dtype=np.float64), (img_w, 1)).T
    return (binary * (x ** a) * (y ** b)).sum()


def u_pq(mus, a, b):
    return mus[a][b] / mus[0][0] ** ((a + b) / 2 + 1)


def hu(binary_image):
    mus = [[mu(binary_image, r, s) for s in range(5)] for r in range(5)]

    hu_moment_1 = u_pq(mus, 2, 0) + u_pq(mus, 0, 2)
    hu_moment_2 = (u_pq(mus, 2, 0) - u_pq(mus, 0, 2)) ** 2 \
                  + 4 * u_pq(mus, 1, 1) ** 2
    hu_moment_3 = (u_pq(mus, 3, 0) - 3 * u_pq(mus, 1, 2)) ** 2 \
                  + (3 * u_pq(mus, 2, 1) - u_pq(mus, 0, 3)) ** 2
    hu_moment_4 = (u_pq(mus, 3, 0) + u_pq(mus, 1, 2)) ** 2 \
                  + (u_pq(mus, 2, 1) + u_pq(mus, 0, 3)) ** 2

    hu_moment_5 = (u_pq(mus, 3, 0) - 3 * u_pq(mus, 1, 2))\
                  * (u_pq(mus, 3, 0) + u_pq(mus, 1, 2))\
                  * ((u_pq(mus, 3, 0) + u_pq(mus, 1, 2)) ** 2 - 3 * (u_pq(mus, 2, 1) + u_pq(mus, 0, 3)) ** 2) \
                  + (3 * u_pq(mus, 2, 1) - u_pq(mus, 0, 3))\
                  * (u_pq(mus, 2, 1) + u_pq(mus, 0, 3))\
                  * (3 * (u_pq(mus, 3, 0) + u_pq(mus, 1, 2)) ** 2 - (u_pq(mus, 2, 1) + u_pq(mus, 0, 3)) ** 2)

    hu_moment_6 = (u_pq(mus, 2, 0) - u_pq(mus, 0, 2)) * \
                  ((u_pq(mus, 3, 0) + u_pq(mus, 1, 2)) ** 2
                   - (u_pq(mus, 2, 1) +
                      u_pq(mus, 0, 3)) ** 2) + 4 * u_pq(mus, 1, 1) \
                  * (u_pq(mus, 3, 0) + u_pq(mus, 1, 2)) * \
                  (u_pq(mus, 2, 1) + u_pq(mus, 0, 3))

    hu_moment_7 = (3 * u_pq(mus, 2, 1) - u_pq(mus, 0, 3)) * \
                  (u_pq(mus, 3, 0) + u_pq(mus, 1, 2)) * \
                  ((u_pq(mus, 3, 0) + u_pq(mus, 1, 2)) ** 2 - 3 * (u_pq(mus, 2, 1) + u_pq(mus, 0, 3)) ** 2) - \
                  (u_pq(mus, 3, 0) - 3 * u_pq(mus, 1, 2)) * (u_pq(mus, 2, 1) + u_pq(mus, 0, 3))\
                  * (3 * (u_pq(mus, 3, 0) + u_pq(mus, 1, 2)) ** 2 - (u_pq(mus, 2, 1) + u_pq(mus, 0, 3)) ** 2)

    return [hu_moment_1, hu_moment_2, hu_moment_3, hu_moment_4, hu_moment_5, hu_moment_6, hu_moment_7]
