import tkinter as tk
import PIL.Image
import PIL.ImageTk
root = tk.Tk()

image_pil = PIL.Image.open('conway_init.png')
W, H = image_pil.size

# tk remark: the variable passed as image in the canvas must be kept around
image_tk = None

canvas = tk.Canvas(root, width=W, height=H)
canvas_image = canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
canvas.pack()

def refresh_image(canvas, canvas_image, i):
  global image_tk
  image_pil = PIL.Image.open(f'out{i}.png')
  image_tk = PIL.ImageTk.PhotoImage(image_pil)
  canvas.itemconfig(canvas_image, image=image_tk)
  canvas.after(10, refresh_image, canvas, canvas_image, i + 1)
refresh_image(canvas, canvas_image, 0)
root.mainloop()