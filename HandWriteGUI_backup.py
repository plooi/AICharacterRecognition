"""
Author: Daniel Leef
"""



from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import Image
import NearestNeighbor
import ImageDataSearch
#Most of this class is from: https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06

training_data = None
num_training_data = 7000


def main():
    global training_data
    
    training_data = ImageDataSearch.get_data("emnist-letters-train.csv", range(1000, num_training_data+1000), "emnist-letters-mapping.txt")
    NearestNeighbor.to_numpy(training_data)
    NearestNeighbor.scale_down(training_data, 12, "average")
    
    p = Paint()
    
class Paint(object):

    DEFAULT_PEN_SIZE = 100.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='Pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.eraser_button = Button(self.root, text='Eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=1)
        
        self.clear_button = Button(self.root, text = 'Clear', command = self.clear)
        self.clear_button.grid(row = 0, column = 2)
        
        self.classify_button = Button(self.root, text = 'Classify', command = self.save_image)
        self.classify_button.grid(row = 0, column = 3)

        self.choose_size_button = Scale(self.root, from_=70, to=70, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=500, height=500)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()
    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def save_image(self):
        self.c.postscript(file = 'input', colormode = 'color')
        self.classify_image()

    def use_pen(self):
        self.activate_button(self.pen_button)
    def clear(self):
        self.c.delete("all")
    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


    def classify_image(self):
        size = 25, 25
        im = Image.open('input')
        im.thumbnail(size, Image.ANTIALIAS)
        
        
        im = im.convert('L')
        width, height = im.size
        pixels = im.load()
        
        #invert, so that white is 0
        for y in range(width):
            for x in range(height):
                pixels[x, y] = 255 - im.getpixel((x,y))
        
        #Executing the nearest neighbor classifier (peter)
        input_data = {"label" : "", "image": im}
        NearestNeighbor.to_numpy([input_data])
        NearestNeighbor.scale_down([input_data], 12, "average")
        n = NearestNeighbor.NearestNeighbor()
        print(n.classify_one(training_data, input_data))
        


if __name__ == '__main__':
    main()
