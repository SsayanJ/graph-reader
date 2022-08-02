import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from logging import Logger
import os

import pytesseract
from pytesseract.pytesseract import Output

from utils import is_number

# Local Tesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# Heroku app
# pytesseract.pytesseract.tesseract_cmd = "/app/.apt/usr/bin/tesseract"

logger = Logger("Graph Reader", level="DEBUG")


class GraphReader:
    def __init__(self) -> None:
        self.source_name = None

    def from_image(
        self,
        image,
        nb_points=100,
        automatic_scale=True,
        x_min_value=None,
        x_max_value=None,
        y_min_value=None,
        y_max_value=None,
    ):
        self.x_min_value = x_min_value
        self.x_max_value = x_max_value
        self.y_min_value = y_min_value
        self.y_max_value = y_max_value
        self.automatic_scale = automatic_scale
        if isinstance(image, str):
            self.source_name = os.path.basename(image)
            self.color_image = cv2.imread(image)
        else:
            self.source_name = "undefined"
            self.color_image = image
        # TODO check if there is a better way to manage this
        self.roi_margin = self.color_image.shape[0] // 100
        self.axis_margin = self.roi_margin * 2
        self.curve_roi, self.x_axis_roi, self.y_axis_roi, self.title_roi = self.get_all_rois()
        self.nb_curves = self.get_curve_number()
        # TODO how to manage the bad quality (colors etc)
        self.mask_curve_roi = self.compute_kmeans(self.curve_roi, self.nb_curves + 1, use_HSV=False)
        self.background_index = self.compute_bg_index()
        self.curves_index = sorted([i for i in np.unique(self.mask_curve_roi) if i != self.background_index])
        self.get_curve_colors()
        if automatic_scale:
            self.match_values_to_ticks()
        self.read_curves(nb_points=nb_points)
        self.scale_results()

    def display_graph(self):
        for x, y, color in zip(self.x_points, self.y_points, self.curves_color):
            # Matplotlib needs the color in RGB between 0-1
            plt.scatter(x, y, color=color / 255, s=4)
        plt.show()

    def get_graph_image(self):
        fig = plt.figure()
        for x, y, color in zip(self.x_points, self.y_points, self.curves_color):
            # Matplotlib needs the color in RGB between 0-1
            plt.scatter(x, y, color=color / 255, s=4)
        # Draw the figure
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def get_curve_colors(self):
        self.curves_color = []
        for curve_id in self.curves_index:
            color = np.mean(self.curve_roi[self.mask_curve_roi == curve_id], axis=0)
            # as OpenCV uses BGR format, the color table needs to be reversed to match RGB used by matplotlib
            self.curves_color.append(color[::-1])

    def detect_axis_ticks(self):
        _, bw_x_axis = cv2.threshold(cv2.cvtColor(self.x_axis_roi, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)
        _, bw_y_axis = cv2.threshold(cv2.cvtColor(self.y_axis_roi, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)
        # check for ticks below the X-axis
        x_ticks = self.get_ticks_indexes(bw_x_axis[int(self.axis_margin * 1.4), :])
        # if not found, check above the X-axis
        if not x_ticks:
            x_ticks = self.get_ticks_indexes(bw_x_axis[int(self.axis_margin * 0.7), :])
            # When ticks are above the axis, it also counts the outside boarder of the chart
            # This fix will be an issue if there is no boarder all around but there will be other issues,
            # for instance curve ROI selection so it is OK for now.
            x_ticks = x_ticks[1: -1]
        if not x_ticks:
            raise RuntimeError("No ticks could be detected around the X-axis, cannot compute the scale of the graph")

        # check for ticks left of the Y-axis
        print("y index", int(self.axis_margin * 1.3))
        y_ticks = self.get_ticks_indexes(bw_y_axis[:, -int(self.axis_margin * 1.4)])
        # if not found, check right of the Y-axis
        if not y_ticks:
            print("not found left")
            y_ticks = self.get_ticks_indexes(bw_y_axis[:, -int(self.axis_margin * 0.7)])
            # Same as for X axis above
            y_ticks = y_ticks[1: -1]
        if not y_ticks:
            raise RuntimeError("No ticks could be detected around the Y-axis, cannot compute the scale of the graph")

        return x_ticks, y_ticks

    def get_ticks_indexes(self, line):
        # Get black pixels on the given line
        positive_matches = np.where(line == 0)[0]
        # print(positive_matches)
        if positive_matches.size > 0:
            # Group consecutive index together as they are the same tick
            grouped_matches = np.split(positive_matches, np.where(np.diff(positive_matches) != 1)[0] + 1)
            # take the mean of each group to get the tick center position
            grouped_matches = [int(np.mean(x)) for x in grouped_matches]
        else:
            grouped_matches = []
        return grouped_matches

    def detect_x_values(self):
        scaling_ratio = 3
        bigger = cv2.resize(self.x_axis_roi, np.array(self.x_axis_roi.shape[:2][::-1]) * scaling_ratio)
        bigger = cv2.erode(bigger, None, iterations=2)
        bigger = cv2.dilate(bigger, None, iterations=1)
        boxes = pytesseract.image_to_data(bigger, output_type=Output.DICT, config="--psm 6 digits")
        x_axis_values, _ = self.process_tesseract_result(boxes, scaling_ratio, y_axis=False)
        return x_axis_values

    def detect_y_values(self):
        scaling_ratio = 3
        bigger = cv2.resize(self.y_axis_roi, np.array(self.y_axis_roi.shape[:2][::-1]) * scaling_ratio)
        # TODO vertical text is an issue, not sure how to remove it. Fixed value here is just working for this example
        bigger = cv2.erode(bigger, None, iterations=2)
        bigger = cv2.dilate(bigger, None, iterations=1)
        boxes = pytesseract.image_to_data(bigger[:, :], output_type=Output.DICT, config="--psm 6")
        y_axis_values, _ = self.process_tesseract_result(boxes, scaling_ratio, y_axis=True)
        return y_axis_values

    def match_values_to_ticks(self):
        x_ticks, y_ticks = self.detect_axis_ticks()
        x_axis_values = self.detect_x_values()
        y_axis_values = self.detect_y_values()
        self.x_axis_matching = {}
        self.y_axis_matching = {}
        # find the closest tick to the center of the number
        for val in x_axis_values:
            self.x_axis_matching[val["value"]] = min(x_ticks, key=lambda x: abs(x - val["center_x"]))
        for val in y_axis_values:
            self.y_axis_matching[val["value"]] = min(y_ticks, key=lambda x: abs(x - val["center_y"]))

    @staticmethod
    def process_tesseract_result(boxes, scaling_ratio, y_axis=False):
        legend = []
        axis_values = []
        right_align = 0

        for txt, left, top, width, height in zip(
            boxes["text"], boxes["left"], boxes["top"], boxes["width"], boxes["height"]
        ):
            if not txt or txt == " ":
                continue
            if is_number(txt):
                # print("axis value", txt)
                axis_values.append(
                    {
                        "value": float(txt),
                        "center_x": (left + width / 2) / scaling_ratio,
                        "center_y": (top + height / 2) / scaling_ratio,
                        "right": left + width,
                    }
                )
                if left + width > right_align:
                    right_align = left + width
            else:
                # print("legend", txt)
                legend.append(
                    {
                        "text": txt,
                        "center_x": (left + width / 2) / scaling_ratio,
                        "center_y": (top + height / 2) / scaling_ratio,
                    }
                )

        # Keep only numbers detected close to the Y-axis
        if y_axis:
            axis_values = [x for x in axis_values if x["right"] > right_align - 10]

        return axis_values, legend

    def compute_bg_index(self):
        index_value_counts = [(self.mask_curve_roi == i).sum() for i in np.unique(self.mask_curve_roi)]
        return np.argmax(index_value_counts)

    @staticmethod
    def get_axis_scale(min_value, max_value, min_pixel, max_pixel):
        logger.debug(f"value min: {min_value} value max: {max_value}")
        logger.debug(f"pixel min: {min_pixel} pixel max: {max_pixel}")
        return abs(max_value - min_value) / abs(max_pixel - min_pixel)

    def get_min_max_values(self):
        self.x_min_value = min(self.x_axis_matching.keys())
        self.x_max_value = max(self.x_axis_matching.keys())
        self.y_min_value = min(self.y_axis_matching.keys())
        self.y_max_value = max(self.y_axis_matching.keys())

    def get_min_max_pixels(self):
        return (
            self.x_axis_matching[self.x_min_value] - self.curve_roi_x_min - self.roi_margin,
            self.x_axis_matching[self.x_max_value] - self.curve_roi_x_min - self.roi_margin,
            self.y_axis_matching[self.y_min_value] - self.curve_roi_y_min - self.roi_margin,
            self.y_axis_matching[self.y_max_value] - self.curve_roi_y_min - self.roi_margin,
        )

    def compute_axis_scales(self):
        # TODO manage different one per curve for mutliple curves
        grey_roi = cv2.cvtColor(self.curve_roi, cv2.COLOR_BGR2GRAY)
        _, black_and_white = cv2.threshold(grey_roi, 200, 255, cv2.THRESH_BINARY)
        binary_roi = cv2.bitwise_not(black_and_white)
        self.x_lower, self.y_lower, width, height = cv2.boundingRect(binary_roi)
        self.x_upper = self.x_lower + width
        self.y_upper = self.y_lower + height

        if self.automatic_scale:
            self.get_min_max_values()
            self.x_min_pixel, self.x_max_pixel, self.y_min_pixel, self.y_max_pixel = self.get_min_max_pixels()
        elif (
            self.x_max_value is None or self.x_min_value is None or self.y_min_value is None or self.y_max_value is None
        ):
            raise RuntimeError(
                "When using manual scale, user needs to provide values for 'x_min_value', 'x_max_value', 'y_min_value',"
                " 'y_max_value'"
            )
        else:
            x_ticks, y_ticks = self.detect_axis_ticks()
            self.x_min_pixel = min(x_ticks) - self.curve_roi_x_min - self.roi_margin
            self.x_max_pixel = max(x_ticks) - self.curve_roi_x_min - self.roi_margin
            self.y_min_pixel = max(y_ticks) - self.curve_roi_y_min - self.roi_margin
            self.y_max_pixel = min(y_ticks) - self.curve_roi_y_min - self.roi_margin

        self.x_scale = self.get_axis_scale(self.x_min_value, self.x_max_value, self.x_min_pixel, self.x_max_pixel)
        self.y_scale = self.get_axis_scale(self.y_min_value, self.y_max_value, self.y_min_pixel, self.y_max_pixel)

    def read_curves(self, nb_points=100):
        self.compute_axis_scales()
        # TODO there are weird effects on the edges of the curves. Check why in more details.
        # Here I just removed some extreme points but it's not a good solution
        self.x_points = [
            np.linspace(self.x_lower + 2, self.x_upper - 3, nb_points) for _ in range(len(self.curves_index))
        ]
        self.y_points = [
            self.read_single_curve(self.mask_curve_roi == curve_id, i) for i, curve_id in enumerate(self.curves_index)
        ]
        self.filter_empty_points()

    def filter_empty_points(self):
        for i in range(len(self.curves_index)):
            mask = self.y_points[i] >= 0
            self.x_points[i] = self.x_points[i][mask]
            self.y_points[i] = self.y_points[i][mask]

    def read_single_curve(self, curve_binary_mask, points_index):
        return np.array(
            [self.get_curve_point_from_binary(image=curve_binary_mask, x=int(x)) for x in self.x_points[points_index]]
        )

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
            self.x_points[i] = [(x - self.x_min_pixel) * self.x_scale + self.x_min_value for x in self.x_points[i]]
            self.y_points[i] = [(self.y_min_pixel - y) * self.y_scale + self.y_min_value for y in self.y_points[i]]

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

        self.curve_roi_x_min, self.curve_roi_y_min = np.min(biggest_rectangle, axis=0)
        self.curve_roi_x_max, self.curve_roi_y_max = np.max(biggest_rectangle, axis=0)

        curve_roi = self.color_image[
            self.curve_roi_y_min + self.roi_margin : self.curve_roi_y_max - self.roi_margin,
            self.curve_roi_x_min + self.roi_margin : self.curve_roi_x_max - self.roi_margin,
        ]
        x_axis_roi = self.color_image[-(self.color_image.shape[0] - self.curve_roi_y_max) - self.axis_margin :, :]
        y_axis_roi = self.color_image[:, : self.curve_roi_x_min + self.axis_margin]
        title_roi = self.color_image[: self.curve_roi_y_min, :]

        return curve_roi, x_axis_roi, y_axis_roi, title_roi

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


if __name__ == "__main__":
    from PIL import Image

    graph_file = "sample_graphs/double_sincos.png"
    graph_file = "sample_graphs/double_tiretee_simple.png"
    automatic_detection = False
    automatic_detection = True
    graph_reader = GraphReader()
    graph = np.array(Image.open(graph_file))
    if automatic_detection:
        graph_reader.from_image(graph, nb_points=100, automatic_scale=True)
        print("automatic ended")
    else:
        graph_reader.from_image(
            graph,
            nb_points=100,
            automatic_scale=False,
            x_min_value=-3,
            x_max_value=3,
            y_min_value=-1,
            y_max_value=1,
        )
        print("manual ended")

# def main_v0(image, x_min_value, x_max_value, y_min_value, y_max_value, nb_points=100):
#     roi = get_curve_roi(image)
#     grey_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     binary_roi = cv2.bitwise_not(grey_roi)
#     x_lower, y_lower, width, height = cv2.boundingRect(binary_roi)
#     x_upper = x_lower + width
#     y_upper = y_lower + height

#     grey_box = grey_roi[y_lower:y_upper, x_lower:x_upper]
#     x_scale = get_axis_scale(x_min_value, x_max_value, x_lower, x_upper)
#     y_scale = get_axis_scale(y_min_value, y_max_value, y_lower, y_upper)
#     # TODO there are weird effects on the edges of the curves. Check why in more details.
#     # Here I just removed some extreme points but it's not a good solution
#     x_points = np.linspace(x_lower + 2, x_upper - 3, nb_points)
#     # print("X points", x_points)
#     logger.debug(f"Scale coefficient X-axis: {x_scale}")
#     logger.debug(f"Scale coefficient Y-axis: {x_scale}")
#     y_points = np.array([get_curve_point(image=grey_roi, x=int(x)) for x in x_points])

#     return (
#         [(x - x_lower) * x_scale + x_min_value for x in x_points],
#         [(y_upper - y) * y_scale + y_min_value for y in y_points],
#         grey_roi,
#         roi,
#     )


# def get_curve_roi(image):
#     """TODO to be removed as it is redundant with get_all_rois"""
#     grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(grey_img, 50, 150, apertureSize=3)
#     # Find contours
#     contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

#     biggest_rectangle = None
#     min_area = image.shape[0] * image.shape[1] / 4
#     for cnt in contours:
#         epsilon = 0.05 * cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)
#         if len(approx) == 4 and cv2.contourArea(cnt) > min_area:
#             biggest_rectangle = np.squeeze(approx)

#     x_min, y_min = np.min(biggest_rectangle, axis=0)
#     x_max, y_max = np.max(biggest_rectangle, axis=0)

#     margin = 3
#     return image[y_min + margin : y_max - margin, x_min + margin : x_max - margin]


# def get_all_rois(image):

#     grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(grey_img, 50, 150, apertureSize=3)
#     # Find contours
#     contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

#     biggest_rectangle = None
#     min_area = image.shape[0] * image.shape[1] / 4
#     for cnt in contours:
#         epsilon = 0.05 * cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)
#         if len(approx) == 4 and cv2.contourArea(cnt) > min_area:
#             biggest_rectangle = np.squeeze(approx)

#     x_min, y_min = np.min(biggest_rectangle, axis=0)
#     x_max, y_max = np.max(biggest_rectangle, axis=0)

#     margin = 3
#     axis_margin = 5
#     curve_roi = image[y_min + margin : y_max - margin, x_min + margin : x_max - margin]
#     x_axis_roi = image[-(image.shape[0] - y_max) - axis_margin :, :]
#     y_axis_roi = image[:, : x_min + axis_margin]

#     return curve_roi, x_axis_roi, y_axis_roi
