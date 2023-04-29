import tkinter as tk
import customtkinter as ctk 

import torch
import torch.hub
from torchvision import datasets, io, models, ops, transforms, utils
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
from PIL import Image, ImageTk
import ctypes

import ctypes
vlc = ctypes.WinDLL(r'C:\Program Files\VideoLAN\VLC\libvlc.dll')
import vlc 
import random

app = tk.Tk()
app.geometry("600x600")
app.title("Drowsy Boi 4.0")
ctk.set_appearance_mode("dark")

vidFrame = tk.Frame(height=480, width=600)
vidFrame.pack()
vid = ctk.CTkLabel(vidFrame)
vid.pack()

counter = 0 

counterLabel = ctk.CTkLabel(app, text=counter,  height=40, width=120, font=("Arial", 20), text_color="white", fg_color="teal")
counterLabel.pack(pady=10)

def reset_counter(): 
    global counter
    counter = 0 

resetButton = ctk.CTkButton(app, text="Reset Counter", command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="teal") 
resetButton.pack()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp9/weights/last.pt', force_reload=True)
cap = cv2.VideoCapture(1)

def detect():
    global counter   # Add this line to reference the global variable `counter`
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error reading frame from camera")
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    img = np.squeeze(results.render())

    if len(results.xywh[0]) > 0: 
        dconf = results.xywh[0][0][4]
        dclass = results.xywh[0][0][5]

        if dconf.item() > 0.85 and dclass.item() == 1.0:
            filechoice = random.choice([1,2,3])
            p = vlc.MediaPlayer(f"file:///{filechoice}.wav")
            p.play() 
            counter += 1 

    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr) 
    vid.imgtk = imgtk
    vid.configure(image=imgtk)
    counterLabel.configure(text=counter)
    vid.after(10, detect)

detect()
app.mainloop()