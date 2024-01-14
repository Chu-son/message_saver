#! /usr/bin/env python3

import os
import numpy as np

from typing import List
import datetime

import pyautogui
import cv2

from saka_message_saver.parameter.parameters import Parameters
from saka_message_saver.different_checker.different_checker import ScrollEndChecker, Criteria, StaticDiffChecker
from saka_message_saver import PROJECT_ROOT_PATH, logger
import saka_message_saver

import time


class ImageSaver:
    def __init__(self, directory: str, filename_base: str):
        self.directory = directory
        self.filename_base = filename_base
        self.index = 0

        self._prepare_directory()

        file_handler = saka_message_saver.FileHandler(os.path.join(directory, 'log.log'))
        file_handler.setLevel(saka_message_saver.DEBUG)
        file_handler.setFormatter(saka_message_saver.formatter)
        logger.addHandler(file_handler)

    def _prepare_directory(self):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

    def save(self, img: np.ndarray):
        img = np.array(img)

        if self.filename_base == '':
            filename_base = self.get_datetime()
        else:
            filename_base = self.filename_base

        filename = os.path.join(
            self.directory, filename_base + "_No" + str(self.index) + '.png')

        cv2.imwrite(filename, img)
        self.index += 1

    @classmethod
    def get_datetime(cls) -> str:
        now = datetime.datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')


class SakaMessageSaver:
    def __init__(self, directory: str, filename_base: str, params: Parameters):
        self.image_saver = ImageSaver(directory, filename_base)
        self.params = params

        if self.params.ROI is None:
            self.params.ROI = self.get_roi()

        # check_area_scale = 1 / 3
        # check_roi = [0, self.params.ROI[3] *
        #              (1 - check_area_scale), self.params.ROI[2], self.params.ROI[3]]
        # check_roi = [int(x) for x in check_roi]
        # self.scroll_end_checker = ScrollEndChecker(roi=check_roi)
        # self.scroll_end_checker = ScrollEndChecker()
        self.scroll_end_checker = ScrollEndChecker(shape=self.params.ROI[2:4],
                                                   threshold=0.005,
                                                   criteria=Criteria.ALL)

        self.load_done_checkers = [
            ScrollEndChecker(shape=self.params.ROI[2:4],
                             threshold=0.01,
                             criteria=Criteria.CENTER,
                             rate=0.15),
            StaticDiffChecker(shape=self.params.ROI[2:4],
                              threshold=0.1,
                              static_img_path=os.path.join(
                                  PROJECT_ROOT_PATH, 'config', 'loading.png'),
                              criteria=Criteria.CENTER,
                              rate=0.15)
        ]

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

        return [roi[0], roi[1], roi[2], roi[3]]

    def _move_mouse_to_scroll_position(self, roi: List[int]):
        pyautogui.moveTo(roi[0] + roi[2] / 2, roi[1] + roi[3] / 2)

    def _move_mouse_to_out_of_roi(self, roi: List[int]):
        pyautogui.moveTo(roi[0] + roi[2], roi[1] + roi[3])

    def _scroll(self, roi: List[int]):
        self._move_mouse_to_scroll_position(roi)
        # pyautogui.sleep(0.15)
        pyautogui.scroll(-10)
        pyautogui.sleep(0.1)
        self._move_mouse_to_out_of_roi(roi)

    def _get_screenshot(self, roi: List[int]) -> np.ndarray:
        # get screen shot
        img = pyautogui.screenshot(region=roi)
        img = np.array(img)

        # convert to numpy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def _wait_for_image_load_done(self, max_try: int = 20):
        # load_done_checker = ScrollEndChecker(shape=self.params.ROI[2:4],
        #                                      criteria=Criteria.CENTER,
        #                                      rate=0.15)

        for _ in range(max_try):
            screenshot = self._get_screenshot(self.params.ROI)
            # if load_done_checker.check(screenshot):
            if all([checker.check(screenshot) for checker in self.load_done_checkers]):
                break
            pyautogui.sleep(0.2)

    def run(self):
        elapsed_time_list = []
        while True:
            logger.info("--------------------")
            logger.info(f"scroll {self.image_saver.index} times.")
            start_time = time.time()

            logger.info('wait for image load done...')
            self._wait_for_image_load_done()
            logger.info('done.')

            screenshot = self._get_screenshot(self.params.ROI)

            self.image_saver.save(screenshot)

            if self.scroll_end_checker.check(screenshot):
                break

            self._scroll(self.params.ROI)
            elapsed_time_list.append(time.time() - start_time)
            logger.info(f"elapsed time: {elapsed_time_list[-1]:.3f} [s]")
            

        logger.info('FINISHED!!')
        logger.info(f"average elapsed time: {np.mean(elapsed_time_list):.3f} [s]")


class SakaMessagePhotoSaver(SakaMessageSaver):
    def __init__(self, directory: str, filename_base: str, params: Parameters):
        super().__init__(directory, filename_base, params)

    # override _scroll method. drag left.
    def _scroll(self, roi: List[int]):
        self._move_mouse_to_scroll_position(roi)
        # pyautogui.sleep(0.15)
        pyautogui.drag(-1000, 0, 0.3, button='left')
        pyautogui.sleep(0.1)
        self._move_mouse_to_out_of_roi(roi)
