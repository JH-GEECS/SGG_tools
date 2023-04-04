from tkinter import *
from tkinterdnd2 import *

def show_text(event):
    textarea.delete("1.0","end")
    if event.data.endswith(".txt"):
        with open(event.data, "r") as file:
            for line in file:
                line=line.strip()
                textarea.insert("end",f"{line}\n")

ws = TkinterDnD.Tk()
ws.title('PythonGuides')
ws.geometry('400x300')
ws.config(bg='#fcb103')

frame = Frame(ws)
frame.pack()

textarea = Text(frame, height=18, width=40)
textarea.pack(side=LEFT)
textarea.drop_target_register(DND_FILES)
textarea.dnd_bind('<<Drop>>', show_text)

sbv = Scrollbar(frame, orient=VERTICAL)
sbv.pack(side=RIGHT, fill=Y)

textarea.configure(yscrollcommand=sbv.set)
sbv.config(command=textarea.yview)



ws.mainloop()