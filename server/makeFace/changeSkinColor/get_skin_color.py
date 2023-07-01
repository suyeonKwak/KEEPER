from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt


def centroid_histogram(km):
    numLabels = np.arange(
        0, len(np.unique(km.labels_)) + 1
    )  # np.unique(km.labels_)를 통해 클러스터 개수를 찾음음
    (hist, _) = np.histogram(km.labels_, bins=numLabels)  # cluster별 원소 개수를 히스토그램으로 만듦

    # Normalize the histogram, such that it sums to one.
    hist = hist.astype("float")
    hist /= hist.sum()

    # Return the histogram
    return hist


def get_color_true(hist, centroids):
    # Obtain the color with maximum percentage of area covered.
    max = 0
    COLOR = [0, 0, 0]
    label = 0

    # Loop over the percentage of each cluster and the colort of each cluster.
    for i, (p, color) in enumerate(zip(hist, centroids)):
        if p > max:
            max = p
            COLOR = color
            label = i

    # Return the most dominant color
    return label, COLOR


def skin(color):
    temp = np.uint8([[color]])
    color = cv2.cvtColor(
        temp, cv2.COLOR_RGB2HSV
    )  # HSV(Hue, Saturation, Value) (색상, 채도, 명도도)
    color = color[0][0]

    e8 = (color[0] <= 25) and (color[0] >= 0)
    e9 = (color[1] < 174) and (color[1] > 58)
    e10 = (color[2] <= 255) and (color[2] > 50)

    return e8 and e9 and e10


def get_color_false(hist, centroids):
    # Obtain a color which satisfies skin condition.
    count = 1
    list = []

    # Loop over the percentage of each cluster and the color of each cluster to see if there is such a color.
    for p, color in zip(hist, centroids):
        # 피부색에 해당되는 색
        if skin(color):
            count += 1
            list.append([color, p])
    if count == 1:
        return list[0][0]
    else:
        return [0, 0, 0]


def get_skin_mask(label, image, shape_0, shape_1):
    image = image.reshape(shape_0, shape_1, 1)
    skin_mask = np.where(image == label, 1, 0)
    # print(skin_mask)

    return skin_mask


def get_skin_color(ori_image):
    # Load the face detector.
    face_cascade = cv2.CascadeClassifier(
        "/content/drive/MyDrive/coders/server_yoojin/makeFace/model/haarcascade_frontalface_alt.xml"
    )

    # crop한 후 원래대로 돌릴 좌표 초기화
    x, y, h, w = 0, 0, 0, 0
    shape_0, shape_1 = 0, 0

    # Convert to grayscale image
    gray = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)

    # Detect face the image(The detected objects are returned as a list of rectangles)
    faces = face_cascade.detectMultiScale(
        gray, 1.3, 5
    )  # image=gray, scaleFactor = 1.3, minNeighbors=5

    # If a face is detected
    if len(faces) > 0:
        # Takce out the face from the image
        x, y, w, h = faces[0]
        crop_image = ori_image[y : y + h, x : x + w]

        # Apply k-Means Clustering to the face to obtain most dominant color.
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)

        shape_0 = crop_image.shape[0]
        shape_1 = crop_image.shape[1]

        re_image = crop_image.reshape((shape_0 * shape_1, 3))  # 가로 세로를 그냥 하나의 차원으로 만들기
        km = KMeans(n_clusters=4)
        km.fit(re_image)

        pred = km.labels_

        # Obtain the color
        hist = centroid_histogram(km)
        label, skin_temp1 = get_color_true(hist, km.cluster_centers_)

        # get skin_mask
        skin_mask = get_skin_mask(label, pred, shape_0, shape_1)
        ori_image_bool = np.zeros((ori_image.shape[0], ori_image.shape[1], 1))
        ori_image_bool[y : y + h, x : x + w] = skin_mask

        # Convert the color to HSV type
        skin_temp2 = np.uint8([[skin_temp1]])
        skin_color = cv2.cvtColor(skin_temp2, cv2.COLOR_RGB2HSV)
        skin_color = skin_color[0][0]

        # Return the color and skin_mask
        return (True, skin_color, ori_image_bool)

    # if a face isn't detected
    else:
        # kmeans 를 이용해 피부색 찾기
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        # 피부, 머리, 기타로 clustering
        km = KMeans(n_clusters=3)
        km.fit(image)

        pred = km.labels_

        hist = centroid_histogram(km)
        label, skin_color = get_color_true(hist, km.cluster_centers_)
        skin_mask = get_skin_mask(label, pred, shape_0, shape_1)
        ori_image_bool = np.zeros(ori_image.shape)
        ori_image_bool[y : y + h, x : x + w] = skin_mask

        if np.array(skin_color).sum() == 0:
            return (False, skin_color, skin_mask)
        else:
            skin_temp2 = np.uint8([[skin_color]])
            skin_color = cv2.cvtColor(skin_temp2, cv2.COLOR_RGB2HSV)
            skin_color = skin_color[0][0]
            return skin_color, skin_mask


# if __name__ == "__main__":
#     image = cv2.imread("C:/Users/user/Desktop/Coders/makeFace/generated/test_v3.png")
#     (find_face, skin_color, skin_mask) = get_skin_color(image)
#     print(skin_mask)
