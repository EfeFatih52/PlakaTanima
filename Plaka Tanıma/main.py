from tkinter import Tk, Button
import os

def open_camera():
    os.system("python KameradanOkuma.py")

def open_photo():
    os.system("python plate.py")


root = Tk()
root.title("Seçim")

camera_btn = Button(root, text="Kamera", command=open_camera)
camera_btn.pack()

photo_btn = Button(root, text="Fotoğraf", command=open_photo)
photo_btn.pack()



root.mainloop()
