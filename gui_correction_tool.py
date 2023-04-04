import glob
import tkinter
import pandas as pd
import cv2
import os
from PIL import ImageTk, Image

import tkinter as tk

class SimpleWindow:
    def __init__(self, csv_dir, img_dir, title="My Window", width=1920, height=1080):
        self.title = title
        self.width = width
        self.height = height

        # data loader 부분
        self.csv_dir = csv_dir
        self.csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
        self.img_dir = img_dir
        self.img_files = glob.glob(os.path.join(img_dir, '*.jpg'))

        self.img_iter = None
        self.current_img = None

        self.pd_iter = None
        self.current_pd = None
        self.current_pd_row_iter = None
        self.current_pd_row = None
        # 중요
        self.iterator_initialize()

        # Create the main window
        self.window = tk.Tk()
        self.window.title(self.title)
        self.window.geometry(f"{self.width}x{self.height}")

        # Add a label to the window
        self.header_label = tk.Label(self.window, text="top1 to top5")
        self.header_label.grid(row=0, column=0)
        self.label = tk.Label(self.window, text="")
        self.label.grid(row=0, column=1)

        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.current_img))
        self.img_label = tk.Label(self.window, image=self.img)
        self.img_label.grid(row=1, column=0)

        # Add a button to the window
        self.button = tk.Button(self.window, text="Click me!", command=self.button_callback)
        self.button.grid(row=0, column=2)

        # Start the main event loop, 프로그램 구동부
        self.window.mainloop()

    def iterator_initialize(self):
        # image caller
        self.img_iter = iter(self.img_files)
        self.current_img = cv2.imread(next(self.img_iter))

        # initial call, bbox iteration과 predicate iteration을 필요로 함
        self.pd_iter = iter(self.csv_files)
        self.current_pd = pd.read_csv(next(self.pd_iter))
        self.current_pd_row_iter = iter(self.current_pd.iterrows())
        self.current_pd_row = next(self.current_pd_row_iter)

    def button_callback(self):
        print("Button clicked!")
        self.label.config(text='Button clicked!')

    def key_callback(self):
        print("Key pressed!")
        self.label.config(text='Key pressed!')

    def new_pd_loader(self):

        pass
    def new_pd_row_loader(self):

        pass

if __name__ == "__main__":
    csv_dir = r'C:\nogada\results'
    image_dir = r'C:\nogada\Image'
    window = SimpleWindow(csv_dir=csv_dir, img_dir=image_dir)