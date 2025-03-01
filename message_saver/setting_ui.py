#! /usr/bin/env python3

import dataclasses

import tkinter as tk
import tkinter.ttk as ttk
from message_saver.parameter.parameters import Parameters

import pyautogui
import cv2
import numpy as np


@dataclasses.dataclass
class WidgetWrapper:
    name: str
    widget: tk.Widget
    side: str


class SettingUI:
    def __init__(self, params: Parameters):
        self.params = params

        self.root = tk.Tk()
        self.root.title('Message Saver Setting')

        self.widgets = []

    def _add_widget(self, name: str, widget: tk.Widget, side: str):
        self.widgets.append(WidgetWrapper(name, widget, side))

    def save(self):
        pass

    def load(self):
        pass

    def close(self):
        self.root.destroy()

    def create_ui(self):
        for param_name, param_value in vars(self.params).items():
            self._add_widget(
                name=f"{param_name}_label",
                widget=ttk.Label(
                    self.root, text=f"{param_name}:{param_value}"
                ),
                side='top'
            )

        self._add_widget(
            name='save_button',
            widget=ttk.Button(
                self.root, text='Save', command=self.save
            ),
            side='bottom'
        )

        self._add_widget(
            name='load_button',
            widget=ttk.Button(
                self.root, text='Load', command=self.load
            ),
            side='bottom'
        )

        self._add_widget(
            name='close_button',
            widget=ttk.Button(
                self.root, text='Close', command=self.close
            ),
            side='bottom'
        )

    def draw(self):
        for widget in self.widgets:
            widget.widget.pack(side=widget.side)

    def get_roi(self) -> list[int]:
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

    def run(self):
        self.params.ROI = self.get_roi()

        # self.create_ui()
        # self.draw()

        # self.root.mainloop()
