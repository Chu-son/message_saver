#! /usr/bin/env python3

import os
import numpy as np

from typing import List
import datetime

import pyautogui
import cv2
import pyaudio
import wave
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

import mss

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
        if self.filename_base == '':
            self.filename_base = self.get_datetime()

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


class MovieSaver(ImageSaver):
    def __init__(self, directory: str, filename_base: str):
        super().__init__(directory, filename_base)

    def save(self, img: np.ndarray, length: float):
        # record screen and audio.
        video_filename = os.path.join(
            self.directory, self.filename_base + "_No" + str(self.index) + '.mp4')
        audio_filename = os.path.join(
            self.directory, self.filename_base + "_No" + str(self.index) + '.wav')

        # video setting
        video_size = (img.shape[1], img.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_frame_rate = 20.0  # frames per second
        out = cv2.VideoWriter(video_filename, fourcc, video_frame_rate,
                              (video_size[0], video_size[1]))

        # audio setting
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        AUDIO_SAMPLE_RATE = 44100  # samples per second
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=AUDIO_SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

        # record
        frames = []
        start_time = time.time()
        video_interval = 1.0 / video_frame_rate  # interval between video frames
        audio_interval = 1.0 / AUDIO_SAMPLE_RATE  # interval between audio samples
        next_video_time = start_time + video_interval
        next_audio_time = start_time + audio_interval
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > length:
                break

            # video
            if current_time >= next_video_time:
                screenshot = pyautogui.screenshot()
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                screenshot = cv2.resize(screenshot, video_size)
                out.write(screenshot)
                next_video_time += video_interval

            # audio
            if current_time >= next_audio_time:
                data = stream.read(CHUNK)
                frames.append(data)
                next_audio_time += audio_interval

        # save
        out.release()
        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open(audio_filename, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(AUDIO_SAMPLE_RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        # combine video and audio
        video_clip = VideoFileClip(video_filename)
        audio_clip = AudioFileClip(audio_filename)
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(os.path.join(
            self.directory, self.filename_base + "_No" + str(self.index) + '_output.mp4'))

        self.index += 1


class SakaMessageSaver:
    def __init__(self, directory: str, filename_base: str, params: Parameters):
        self.image_saver = ImageSaver(directory, filename_base)
        self.params = params

        if self.params.ROI is None:
            self.params.ROI = self.get_roi()

        self.scroll_end_checker = ScrollEndChecker(shape=self.params.ROI[2:4],
                                                   threshold=0.005,
                                                   criteria=Criteria.ALL)

        self.load_done_checkers = [
            ScrollEndChecker(shape=self.params.ROI[2:4],
                             threshold=0.005,
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
        # get screen shot by mss
        monitor = {"top": roi[1], "left": roi[0],
                     "width": roi[2], "height": roi[3]}
        with mss.mss() as sct:
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

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
        start_time_total = time.time()
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
        logger.info(
            f"average elapsed time: {np.mean(elapsed_time_list):.3f} [s]")
        logger.info(
            f"total time: {time.time() - start_time_total:.3f} [s]")


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


class SakaMessageMovieSaver(SakaMessageSaver):
    def __init__(self, directory: str, filename_base: str, params: Parameters):
        super().__init__(directory, filename_base, params)

    # override _scroll method. drag left.
    def _scroll(self, roi: List[int]):
        self._move_mouse_to_scroll_position(roi)
        # pyautogui.sleep(0.15)
        pyautogui.drag(-1000, 0, 0.3, button='left')
        pyautogui.sleep(0.1)
        self._move_mouse_to_out_of_roi(roi)

    def _get_movie_length(self, roi: List[int]) -> int:
        # get screen shot
        img = pyautogui.screenshot(region=roi)
        img = np.array(img)

        # convert to numpy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img.shape[1]

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
        logger.info(
            f"average elapsed time: {np.mean(elapsed_time_list):.3f} [s]")
