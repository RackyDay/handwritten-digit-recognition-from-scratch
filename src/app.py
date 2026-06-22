import tkinter as tk
from model import load_model, predict
import numpy as np

root = tk.Tk()
root.title("DigitGuessr")
weights, biases = load_model("model.npz")

canvas_x, canvas_y = 280, 280
canvas = tk.Canvas(root, width = canvas_x, height = canvas_y, bg="black")
canvas.pack()

brush_radius = tk.IntVar(value=8)
pixels = np.zeros((canvas_x, canvas_y))

predict_job = None

def paint(event):
    global predict_job
    radius = brush_radius.get()
    x, y = event.x, event.y

    canvas.create_oval(
        x - radius, y - radius,
        x + radius, y + radius,
        fill = "white",
        outline="white"
    )

    pixels[max(0, y - radius):y + radius, max(0, x - radius): x+radius] = 1.0

    if predict_job is not None:
        root.after_cancel(predict_job)
    
    predict_job = root.after(250, make_prediction)

def clear():
    canvas.delete("all")
    pixels[:] = 0

def make_prediction():
    small = pixels.reshape(28, 10, 28, 10).max(axis=(1, 3))
    image = small.flatten()
    prediction = predict(image, weights, biases)
    
    print(np.argmax(prediction))

canvas.bind("<B1-Motion>", paint)

frame = tk.Frame(root)
frame.pack()

tk.Label(frame, text="Brush Radius: ").grid(row=0, column = 0)
tk.Scale(frame, from_=1, to=30, orient="horizontal", variable=brush_radius).grid(row=0, column=1)
tk.Button(frame, text="Clear", command=clear).grid(row=0, column=2)

root.mainloop()
