import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import csv
from tkinter import ttk
import tkinterdnd2 as tkdnd2

import shutil

"""
1. 특정 csv file을 gui에 drag and drop을 한다.
2. excel을 통해서 해당 file을 open을 해준다.
3. 해당 file과 같은 file 명을 가진 iamge file을 open을 해준다.
4. 그리고 특정 md file에 해당 file 명을 기입을 해준다.
5. 마지막으로 save button을 누르면, 해당 csv file을 다른 경로에 저장을 해준다.
6. 지금까지 open한 csv file들의 수를 count한다. 

"""

class FileOpener(tkdnd2.TkinterDnD.Tk):
    def __init__(self, root_dir, sub_dirs, report_file_path):
        super().__init__()
        self.geometry("1920x1080")
        self.title("File Opener")

        self.root_dir = root_dir
        self.sub_dirs = sub_dirs
        self.report_file_path = report_file_path
        self.cur_file_path = None
        self.cur_file_name = None
        self.current_processed_file = 0

        self.save_button = tk.Button(self, text="Save File", command=self.save_file)
        self.save_button.pack()

        self.cur_processed_file_display = tk.Label(self)
        self.cur_processed_file_display.pack()
        self.cur_processed_file_display.configure(text=f"Current processed files: {self.current_processed_file}")

        self.prev_file_path = None
        self.del_button = tk.Button(self, text="Delete prev File", command=self.del_file)
        self.del_button.pack()

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=20)
        self.image_label.configure(text="drop a file here to open")
        self.image_label.drop_target_register(tkdnd2.DND_FILES)
        self.image_label.dnd_bind("<<Drop>>", self.open_file)
        self.image_label.dnd_bind("<<DragEnter>>", lambda event: self.image_label.configure(text="Drop to open"))

    def del_file(self):
        os.remove(self.prev_file_path)
        self.prev_file_path = None


    def open_file(self, event):
        file_path = event.data
        self.cur_file_path = file_path
        os.path.basename(file_path)
        file_name = os.path.basename(file_path).split('.')[0]
        self.cur_file_name = file_name
        image_path = os.path.join(self.root_dir, self.sub_dirs[0], file_name + '.jpg')

        try:
            raw_image = Image.open(image_path)
            # 크기 조절 필요
            raw_image = raw_image.resize((raw_image.width, raw_image.height))
            image = ImageTk.PhotoImage(raw_image)
            self.image_label.configure(image=image)
            self.image_label.image = image
        except:
            self.image_label.configure(text="No image found")

        subprocess.Popen([r'C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE', file_path])

    def save_file(self):
        # label된 file임을 의미하기 위하여
        with open(self.report_file_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.cur_file_name, True])

        # label data를 저장하기 위하여
        shutil.copy(self.cur_file_path, os.path.join(self.root_dir, self.sub_dirs[1], self.cur_file_name + '.csv'))

        # statistics
        self.prev_file_path = self.cur_file_path
        self.cur_file_path = None
        self.cur_file_name = None
        self.current_processed_file += 1
        self.cur_processed_file_display.configure(text=f"Current processed files: {self.current_processed_file}")

if __name__ == "__main__":
    root_dir = r'C:\model_temp\dataset'
    # 첫번째 경로는 이미지 경로, 두번째 경로는 train data set으로 사용할 부분
    sub_dirs = ['annotated_imgs-use', 'prometus-use-2']
    report_file_path = r'C:\model_temp\dataset\progress_report\prometeus-use-2.csv'
    app = FileOpener(root_dir, sub_dirs, report_file_path)
    app.mainloop()