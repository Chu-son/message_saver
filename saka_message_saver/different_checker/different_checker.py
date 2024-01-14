import os
import numpy as np
import enum
import cv2

from typing import List


class Criteria(enum.IntEnum):
    TOP = enum.auto()
    BOTTOM = enum.auto()
    CENTER = enum.auto()
    ALL = enum.auto()


class ScrollEndChecker:
    DEBUG = False

    # shape: [width, height]
    def __init__(self, shape: List[int], threshold: float,
                 criteria: Criteria, rate: float = 0.1):
        self.roi = self.calculate_roi(shape, criteria, rate)
        self.threshold = threshold
        self.prev_img = None

    # return roi: [x, y, width, height]
    def calculate_roi(self, shape: List[int], criteria: Criteria, rate: float) -> List[int]:
        if criteria == Criteria.TOP:
            return [0, 0, shape[0], int(shape[1] * rate)]
        elif criteria == Criteria.BOTTOM:
            return [0, int(shape[1] * (1 - rate)), shape[0], int(shape[1] * rate)]
        elif criteria == Criteria.CENTER:
            return [0, int(shape[1] / 2 - shape[1] * rate / 2), shape[0], int(shape[1] * rate)]
        elif criteria == Criteria.ALL:
            return [0, 0, shape[0], shape[1]]
        else:
            raise ValueError(
                f'criteria is invalid value. criteria: {criteria}')

    def _pre_process(self, img: np.ndarray) -> np.ndarray:
        # roi: [x, y, width, height]
        img = img[self.roi[1]:self.roi[1] + self.roi[3],
                  self.roi[0]:self.roi[0] + self.roi[2]]
        return img

    def _calculate_diff(self, img: np.ndarray) -> float:
        # check image difference.
        diff = cv2.absdiff(self.prev_img, img)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff = diff.astype(np.float32)
        diff = diff / 255.0
        diff = np.mean(diff)

        return diff

    def _evaluate(self, diff: float, threshold: float) -> bool:
        if diff < threshold:
            return True

        return False

    def _post_process(self, img: np.ndarray) -> np.ndarray:
        self.prev_img = img

        return img

    def _visualize(self, img: np.ndarray):
        cv2.namedWindow('prev image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('check image', cv2.WINDOW_NORMAL)
        cv2.imshow('prev image', self.prev_img)
        cv2.imshow('check image', img)
        cv2.waitKey(1)

    # if return True, scroll end.
    def check(self, img: np.ndarray) -> bool:
        img = np.array(img)

        img = self._pre_process(img)

        if self.prev_img is None:
            self.prev_img = img
            return False

        # show image for debug
        if self.DEBUG:
            self._visualize(img)

        # check image difference.
        diff = self._calculate_diff(img)
        print(f"class: {self.__class__.__name__}. diff: {diff:.5f}. threshold: {self.threshold:.5f}. evaluate: {self._evaluate(diff, self.threshold)}")

        self._post_process(img)

        return self._evaluate(diff, self.threshold)


class StaticDiffChecker(ScrollEndChecker):
    def __init__(self, shape: List[int], static_img_path: str, threshold: float,
                 criteria: Criteria, rate: float = 0.1):
        super().__init__(shape, threshold, criteria, rate)

        # check path exists
        if not os.path.exists(static_img_path):
            raise FileNotFoundError(f'{static_img_path} is not found.')

        self.prev_img = cv2.imread(static_img_path)
        if self.prev_img is None:
            raise ValueError(f'{static_img_path} is not image file.')

        if self.prev_img.shape[0] != shape[1] or self.prev_img.shape[1] != shape[0]:
            self.prev_img = cv2.resize(self.prev_img, (shape[0], shape[1]))

        self.prev_img = self._pre_process(self.prev_img)

    def _post_process(self, img: np.ndarray) -> np.ndarray:
        return img

    def _evaluate(self, diff: float, threshold: float) -> bool:
        return not super()._evaluate(diff, threshold)
