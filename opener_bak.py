import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import csv
from tkinter import ttk

"""
1. 특정 csv file을 gui에 drag and drop을 한다.
2. excel을 통해서 해당 file을 open을 해준다.
3. 해당 file과 같은 file 명을 가진 iamge file을 open을 해준다.
4. 그리고 특정 md file에 해당 file 명을 기입을 해준다.
5. 마지막으로 save button을 누르면, 해당 csv file을 다른 경로에 저장을 해준다.
6. 지금까지 open한 csv file들의 수를 count한다. 

"""

class Opener(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)

        self.master = master

        self.root_dir = r'C:\model_temp\dataset'

        self.sub_dirs = ['input_image_raw_trim-use',
                         'prometeus-egg1_bak']
        self.report_file = r'C:\model_temp\dataset\progress_report\prometeus-egg1_bak.csv'

        self.master.bind("<<Drop>>", self.handle_drop)
        self.master.bind("<<DragEnter>>", self.handle_drag_enter)
        self.master.bind("<<DragLeave>>", self.handle_drag_leave)

        self.drop_target = None
        self.drag_enter_counter = 0

        self.cur_file_name = None

    def create_widgets(self):
        # create the "Save File" button
        self.save_button = tk.Button(self, text="Save File", command=self.save_file)
        self.save_button.pack(side="bottom")

        # create the image widget
        self.image = tk.Label(self)
        self.image.pack(side="top")

    def handle_drag_enter(self, event):
        # Highlight the drop target when a file is dragged onto the GUI
        self.drag_enter_counter += 1
        if self.drag_enter_counter == 1:
            self.drop_target = tk.Frame(self.master, width=self.master.winfo_width(), height=self.master.winfo_height(),
                                        bg="blue")
            self.drop_target.place(x=0, y=0)

    def handle_drag_leave(self, event):
        # Unhighlight the drop target when a file is dragged off the GUI
        self.drag_enter_counter -= 1
        if self.drag_enter_counter == 0:
            self.drop_target.destroy()
            self.drop_target = None

    # 이 부분에서 exel 다루는 것이랑,image open이랑 변형을 해야한다.
    def handle_drop(self, event):
        # Open the dropped file with Excel and try to display the corresponding image
        file_path = event.data.strip()
        self.cur_file_name = file_path.split('\\')[-1]
        file_name = file_path.split('\\')[-1].splt('.')[0] # 이름만 가져오기
        if os.path.isfile(file_path):
            subprocess.call(['open', '-a', 'Microsoft Excel.app', file_path])
            file_name = os.path.join(self.root_dir, self.sub_dirs[0], file_name+'.jpg')
            try:
                image_path = f"{file_name}.jpg"  # Assumes the image has the same name as the file, but with a .jpg extension
                image = Image.open(image_path)
                # image = image.resize((200, 200))  # Resize the image, 이미지가 작아서 원본으로 열려야함
                photo = ImageTk.PhotoImage(image)
                self.image.configure(image=photo)
                self.image.image = photo
            except FileNotFoundError:
                print(f"No image found for file {file_name}")

    # 이 부분도 조금만 수정하면 될 듯 하다.
    def save_file(self):
        # 내가 해당 file 수정하였는 가를 기입하기 위함
        # Save the file (here we just create an empty file)
        with open(self.report_file, "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.cur_file_name, True])
        self.cur_file_name = None

if __name__ == "__main__":
    master = tk.Tk()
    app = Opener(master)
    app.mainloop()