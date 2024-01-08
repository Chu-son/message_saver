#! /usr/bin/env python3

import os
import numpy as np
import pyautogui
import cv2
from typing import List


class ScrollEndChecker:
    def __init__(self, roi: List[int] = None):
        self.roi = roi
        self.prev_img = None

    def check(self, img: np.ndarray) -> bool:
        img = np.array(img)

        if self.roi is None:
            self.roi = [0, 0, img.shape[1], img.shape[0]]

        img = img[self.roi[1]:self.roi[1] + self.roi[3],
                  self.roi[0]:self.roi[0] + self.roi[2]]

        if self.prev_img is None:
            self.prev_img = img
            return False

        # show image for debug
        cv2.namedWindow('check image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('prev image', cv2.WINDOW_NORMAL)
        cv2.imshow('check image', img)
        cv2.imshow('prev image', self.prev_img)
        cv2.waitKey(1)

        # if np.array_equal(self.prev_img, img):
            # return True
        
        # check image difference. 95% of pixels are same, then return True
        diff = cv2.absdiff(self.prev_img, img)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff = diff.astype(np.float32)
        diff = diff / 255.0
        diff = np.mean(diff)
        print(diff)
        if diff < 0.05:
            return True

        self.prev_img = img
        return False


class ImageSaver:
    def __init__(self, directory: str, filename_base: str):
        self.directory = directory
        self.filename_base = filename_base
        self.index = 0

        self._prepare_directory()

    def _prepare_directory(self):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def save(self, img: np.ndarray):
        img = np.array(img)

        filename = os.path.join(
            self.directory, self.filename_base + str(self.index) + '.png')

        cv2.imwrite(filename, img)
        self.index += 1


class SakaMessageSaver:
    def __init__(self, directory: str, filename_base: str):
        self.image_saver = ImageSaver(directory, filename_base)

    def get_roi(self) -> List[int]:
        # get screen shot
        img = pyautogui.screenshot()
        img = np.array(img)

        # convert to numpy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # show image
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)

        # get roi by user input of mouse click event
        roi = cv2.selectROI('image', img, False)
        cv2.destroyAllWindows()

        # crop and draw
        cropped_img = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        cv2.namedWindow('cropped image', cv2.WINDOW_NORMAL)
        cv2.imshow('cropped image', cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return roi[0], roi[1], roi[2], roi[3]

    def _get_screenshot(self, roi: List[int]) -> np.ndarray:
        # get screen shot
        img = pyautogui.screenshot(region=roi)
        img = np.array(img)

        # convert to numpy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def run(self):
        # get roi
        roi = self.get_roi()

        check_area_scale = 1 / 3
        check_roi = [0, roi[3] * (1 - check_area_scale), roi[2], roi[3]]
        check_roi = [int(x) for x in check_roi]
        scroll_end_checker = ScrollEndChecker(roi=check_roi)

        # mouse move to center of roi
        pyautogui.moveTo(roi[0] + roi[2] / 2, roi[1] + roi[3] / 2)

        # scroll
        while True:
            screenshot = self._get_screenshot(roi)

            self.image_saver.save(screenshot)

            if scroll_end_checker.check(screenshot):
                break

            pyautogui.moveTo(roi[0] + roi[2] / 2, roi[1] + roi[3] / 2)
            pyautogui.scroll(-10)
            pyautogui.sleep(1)

        print('scroll end')


if __name__ == '__main__':
    saver = SakaMessageSaver(directory='images/test',
                             filename_base='image')
    saver.run()
