import cv2
import matplotlib.pyplot as plt
import numpy as np
from pandas import value_counts
from sklearn.cluster import KMeans
from logging import Logger
import os

logger = Logger("Graph Reader", level="DEBUG")


class GraphReader:
    def __init__(self) -> None:
        self.source_name = None

    def from_image(self, image_path, nb_points=100):
        self.source_name = os.path.basename(image_path)
        self.color_image = cv2.imread(image_path)
        self.curve_roi, self.x_axis_roi, self.y_axis_roi = self.get_all_rois()
        self.nb_curves = self.get_curve_number()
        self.mask_curve_roi = self.compute_kmeans(self.curve_roi, self.nb_curves + 1, use_HSV=False)
        self.background_index = self.compute_bg_index()
        self.curves_index = sorted([i for i in np.unique(self.mask_curve_roi) if i != self.background_index])
        self.read_curves(self.mask_curve_roi, nb_points)
        self.scale_results()

    def compute_bg_index(self):
        index_value_counts = [(self.mask_curve_roi == i).sum() for i in np.unique(self.mask_curve_roi)]
        return np.argmax(index_value_counts)

    @staticmethod
    def get_axis_scale(min_value, max_value, min_pixel, max_pixel):
        logger.debug(f"value min: {min_value} value max: {max_value}")
        logger.debug(f"pixel min: {min_pixel} pixel max: {max_pixel}")
        return (max_value - min_value) / (max_pixel - min_pixel)

    def compute_axis_scales(self, x_min_value, x_max_value, y_min_value, y_max_value):
        grey_roi = cv2.cvtColor(self.curve_roi, cv2.COLOR_BGR2GRAY)
        _, black_and_white = cv2.threshold(grey_roi, 200, 255, cv2.THRESH_BINARY)
        binary_roi = cv2.bitwise_not(black_and_white)
        self.x_lower, self.y_lower, width, height = cv2.boundingRect(binary_roi)
        self.x_upper = self.x_lower + width
        self.y_upper = self.y_lower + height
        self.x_scale = self.get_axis_scale(x_min_value, x_max_value, self.x_lower, self.x_upper)
        self.y_scale = self.get_axis_scale(y_min_value, y_max_value, self.y_lower, self.y_upper)

    def read_curves(self, mask_curve_roi, nb_points=100):
        self.x_min_value, self.x_max_value = -np.pi, np.pi
        self.y_min_value, self.y_max_value = -1, 1
        self.compute_axis_scales(-np.pi, np.pi, -1, 1)
        # TODO there are weird effects on the edges of the curves. Check why in more details.
        # Here I just removed some extreme points but it's not a good solution
        self.x_points = [np.linspace(self.x_lower + 2, self.x_upper - 3, nb_points) for _ in range(len(self.curves_index))]
        self.y_points = [self.read_single_curve(self.mask_curve_roi == curve_id, i) for i, curve_id in enumerate(self.curves_index)]
        self.filter_empty_points()

    def filter_empty_points(self):
        for i in range(len(self.curves_index)):
            mask = self.y_points[i] >= 0
            self.x_points[i] = self.x_points[i][mask]
            self.y_points[i] = self.y_points[i][mask]

    def read_single_curve(self, curve_binary_mask, points_index):
        return np.array([self.get_curve_point_from_binary(image=curve_binary_mask, x=int(x)) for x in self.x_points[points_index]])

    @staticmethod
    def get_curve_point_from_binary(image, x):
        min_indexes = np.where(image[:, x] == 1)
        # Check if there is at least 1 value that matches or return -1
        if min_indexes[0].size > 0:
            return int(np.mean(min_indexes))
        else:
            return -1

    def scale_results(self):
        for i in range(len(self.x_points)):
            self.x_points[i] = [(x - self.x_lower) * self.x_scale + self.x_min_value for x in self.x_points[i]]
            self.y_points[i] = [(self.y_upper - y) * self.y_scale + self.y_min_value for y in self.y_points[i]]

    def get_all_rois(self):

        grey_img = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey_img, 50, 150, apertureSize=3)
        # Find contours
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        biggest_rectangle = None
        min_area = self.color_image.shape[0] * self.color_image.shape[1] / 4
        for cnt in contours:
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4 and cv2.contourArea(cnt) > min_area:
                biggest_rectangle = np.squeeze(approx)

        x_min, y_min = np.min(biggest_rectangle, axis=0)
        x_max, y_max = np.max(biggest_rectangle, axis=0)

        margin = 3
        axis_margin = 5
        curve_roi = self.color_image[y_min + margin : y_max - margin, x_min + margin : x_max - margin]
        x_axis_roi = self.color_image[-(self.color_image.shape[0] - y_max) - axis_margin :, :]
        y_axis_roi = self.color_image[:, : x_min + axis_margin]

        return curve_roi, x_axis_roi, y_axis_roi

    @staticmethod
    def compute_kmeans(curve_roi, nb_clusters, use_HSV=False):
        if use_HSV:
            curve_roi = cv2.cvtColor(curve_roi, cv2.COLOR_BGR2HSV)
        X = curve_roi.reshape(curve_roi.shape[0] * curve_roi.shape[1], curve_roi.shape[2])
        result = KMeans(n_clusters=nb_clusters).fit_predict(X)
        return result.reshape(curve_roi.shape[:2])

    @staticmethod
    def count_curves(gray_roi, x):
        current_pixel = 255
        nb_curves = 0
        for pixel in gray_roi[:, x]:
            if current_pixel > 250 and pixel < 240:
                nb_curves += 1
            current_pixel = pixel
        logger.debug("x", x, "nb_curves:", nb_curves)
        return nb_curves

    def get_curve_number(self):
        grey_roi = cv2.cvtColor(self.curve_roi, cv2.COLOR_BGR2GRAY)
        return max([self.count_curves(grey_roi, int(i * grey_roi.shape[1] / 6)) for i in range(2, 6)])


def get_curve_point(image, x):
    min_indexes = np.where(image[:, x] == np.min(image[:, x]))
    return int(np.mean(min_indexes))


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
    # TODO there are weird effects on the edges of the curves. Check why in more details.
    # Here I just removed some extreme points but it's not a good solution
    x_points = np.linspace(x_lower + 2, x_upper - 3, nb_points)
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
    x_axis_roi = image[-(image.shape[0] - y_max) - axis_margin :, :]
    y_axis_roi = image[:, : x_min + axis_margin]

    return curve_roi, x_axis_roi, y_axis_roi
