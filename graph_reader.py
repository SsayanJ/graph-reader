from cgitb import grey
from distutils.log import Log
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from logging import Logger
import os

logger = Logger("Graph Reader", level="DEBUG")


class GraphReader():
    def __init__(self) -> None:
        self.source_name = None

    def from_image(self, image_path, nb_points=100):
        self.source_name = os.path.basename(image_path)
        self.color_image = cv2.imread(image_path)
        self.curve_roi, self.x_axis_roi, self.y_axis_roi = self.get_all_rois(self.color_image)
        self.nb_curves = self.get_curve_number(self.curve_roi)
        self.mask_curve_roi = self.compute_kmeans(self.curve_roi, self.nb_curves+1, use_HSV=True)
        self.curve_points = self.read_curves(self.mask_curve_roi, nb_points)
    
    def read_curves(self, mask_curve_roi, nb_points=100):
        curve_points = None
        return curve_points 
    
    def read_single_curve(self, curve_binary_mask, nb_points=100):
        
        return x_points, y_points
    
    def get_curve_point_from_binary(image, x):
        min_indexes = np.where(image[:, x] == 1)
        return int(np.mean(min_indexes))

    def get_all_rois(image):

        grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey_img, 50, 150, apertureSize=3)
        # Find contours
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        biggest_rectangle = None
        min_area = image.shape[0] * image.shape[1] / 4
        for cnt in contours:
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4 and cv2.contourArea(cnt) > min_area:
                biggest_rectangle = np.squeeze(approx)

        x_min, y_min = np.min(biggest_rectangle, axis=0)
        x_max, y_max = np.max(biggest_rectangle, axis=0)

        margin = 3
        axis_margin = 5
        curve_roi = image[y_min + margin : y_max - margin, x_min + margin : x_max - margin]
        x_axis_roi = image[-(image.shape[0]-y_max) - axis_margin :, :]
        y_axis_roi = image[:, : x_min + axis_margin]

        return curve_roi, x_axis_roi, y_axis_roi
    
    def compute_kmeans(curve_roi, nb_clusters, use_HSV=True):
        if use_HSV:
            curve_roi = cv2.cvtColor(curve_roi, cv2.COLOR_BGR2HSV)
        X = curve_roi.reshape(curve_roi.shape[0] * curve_roi.shape[1], curve_roi.shape[2])
        result = KMeans(n_clusters=nb_clusters).fit_predict(X)
        return result.reshape(curve_roi.shape[:2])


    def count_curves(gray_roi, x):
        current_pixel = 255
        nb_curves = 0
        for pixel in gray_roi[:, x]:
            if current_pixel > 250 and pixel < 240:
                nb_curves += 1
            current_pixel = pixel
        logger.debug("x", x, "nb_curves:", nb_curves)
        return nb_curves

    def get_curve_number(self, curve_roi):
        grey_roi = cv2.cvtColor(curve_roi, cv2.COLOR_BGR2GRAY)
        return max([self.count_curves(grey_roi, int(i * grey_roi.shape[1] / 6)) for i in range(2, 6)])


def get_curve_point(image, x):
    min_indexes = np.where(image[:, x] == np.min(image[:, x]))
    return int(np.mean(min_indexes))


def get_axis_scale(min_value, max_value, min_pixel, max_pixel):
    logger.debug(f"value min: {min_value} value max: {max_value}")
    logger.debug(f"pixel min: {min_pixel} pixel max: {max_pixel}")
    return (max_value - min_value) / (max_pixel - min_pixel)


def main_v0(image, x_min_value, x_max_value, y_min_value, y_max_value, nb_points=100):
    roi = get_curve_roi(image)
    grey_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary_roi = cv2.bitwise_not(grey_roi)
    x_lower, y_lower, width, height = cv2.boundingRect(binary_roi)
    x_upper = x_lower + width
    y_upper = y_lower + height

    grey_box = grey_roi[y_lower:y_upper, x_lower:x_upper]
    x_scale = get_axis_scale(x_min_value, x_max_value, x_lower, x_upper)
    y_scale = get_axis_scale(y_min_value, y_max_value, y_lower, y_upper)
    x_points = np.linspace(x_lower, x_upper - 1, nb_points)
    # print("X points", x_points)
    logger.debug(f"Scale coefficient X-axis: {x_scale}")
    logger.debug(f"Scale coefficient Y-axis: {x_scale}")
    y_points = np.array([get_curve_point(image=grey_roi, x=int(x)) for x in x_points])

    return (
        [(x - x_lower) * x_scale + x_min_value for x in x_points],
        [(y_upper - y) * y_scale + y_min_value for y in y_points],
        grey_roi,
        roi,
    )


def get_curve_roi(image):
    """TODO to be removed as it is redundant with get_all_rois"""
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey_img, 50, 150, apertureSize=3)
    # Find contours
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    biggest_rectangle = None
    min_area = image.shape[0] * image.shape[1] / 4
    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > min_area:
            biggest_rectangle = np.squeeze(approx)

    x_min, y_min = np.min(biggest_rectangle, axis=0)
    x_max, y_max = np.max(biggest_rectangle, axis=0)

    margin = 3
    return image[y_min + margin : y_max - margin, x_min + margin : x_max - margin]


def get_all_rois(image):

    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey_img, 50, 150, apertureSize=3)
    # Find contours
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    biggest_rectangle = None
    min_area = image.shape[0] * image.shape[1] / 4
    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > min_area:
            biggest_rectangle = np.squeeze(approx)

    x_min, y_min = np.min(biggest_rectangle, axis=0)
    x_max, y_max = np.max(biggest_rectangle, axis=0)

    margin = 3
    axis_margin = 5
    curve_roi = image[y_min + margin : y_max - margin, x_min + margin : x_max - margin]
    x_axis_roi = image[-(image.shape[0]-y_max) - axis_margin :, :]
    y_axis_roi = image[:, : x_min + axis_margin]

    return curve_roi, x_axis_roi, y_axis_roi



