from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image, ImageFilter
import numpy as np

model = load_model('model.h5')


def prepare_image(img):
    img.filter(ImageFilter.SHARPEN)
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img/255.0
    return img


def predict_digit(img):
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(
            self, width=300, height=300, bg="black", cursor="cross")
        self.label = tk.Label(self, text="Draw...", font=("Helvetica", 48))
        self.classify_btn = tk.Button(
            self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(
            self, text="Clear", command=self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        a = self.canvas.winfo_rootx()
        b = self.canvas.winfo_rooty()
        c, d = a + self.canvas.winfo_width(), b + self.canvas.winfo_height()
        rect = (a+4, b+4, c-4, d-4)
        im = ImageGrab.grab(rect)
        prepared_img = prepare_image(im)

        digit, acc = predict_digit(prepared_img)
        acc = int(acc*100)
        self.label.configure(text=f"{digit} , {acc} %")

    def draw(self, event):
        self.x = event.x
        self.y = event.y
        r = 10
        self.canvas.create_oval(
            self.x-r, self.y-r, self.x+r, self.y+r, fill="white")


app = App()
app.mainloop()
