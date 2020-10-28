"""
Author: Daniel Leef
"""

from tkinter import *
from tkinter.colorchooser import askcolor
import tkinter.font as tkFont
from PIL import Image
import NearestNeighbor
import ImageDataSearch

#Some of this is borrowed from: https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06

training_data = None
num_training_data = 30000
canvasWidth = 500
canvasHeight = 500

def main():
    print("This may take 1-2 minutes, due to the high quantity of data that must be loaded in. Thank you for your patience.")
    global training_data
    
    training_data = ImageDataSearch.get_data("emnist-letters-train.csv", range(1000, num_training_data+1000), "emnist-letters-mapping.txt")
    NearestNeighbor.trim_whitespace(training_data)
    NearestNeighbor.to_numpy(training_data)
    NearestNeighbor.scale_down(training_data, 12, "average")
    
    p = Draw()
    
class Draw(object):

    def __init__(self):
        self.root = Tk()

        self.draw = Button(self.root, text='Draw', command=self.useDraw)
        self.draw.grid(row=0, column=0)

        self.eraser = Button(self.root, text='Erase', command=self.useEraser)
        self.eraser.grid(row=0, column=1)
        
        self.clearButton = Button(self.root, text = 'Clear', command = self.clear)
        self.clearButton.grid(row = 0, column = 2)

        self.classifyButton = Button(self.root, text = 'Classify', command = self.saveImage)
        self.classifyButton.grid(row = 0, column = 4)

        self.c = Canvas(self.root, bg='white', width=canvasWidth, height=canvasHeight)
        self.c.grid(row=1, columnspan=5)
        
        self.setup()
        self.root.mainloop()
    
    def setup(self):
        self.oldX= None
        self.oldY = None
        self.color = 'black'
        self.activeButton = self.draw
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.topExists = False

    def useDraw(self):
        self.activeButton= self.draw

    def useEraser(self):
        self.activeButton= self.eraser

    def saveImage(self):
        self.c.postscript(file = 'input', colormode = 'color')
        self.top = Toplevel()
        self.top.title("Classification")
        self.topExists = True
        self.classifyImage()

    def clear(self):
        self.c.delete("all")
        if (self.topExists):
            self.top.destroy()
            self.topExists == False

    def paint(self, event):
        if (self.activeButton == self.eraser):
            color = 'white'
        else:
            color = self.color
        if self.oldX and self.oldY:
            self.c.create_line(self.oldX, self.oldY, event.x, event.y,
                               width=70, fill=color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.oldX = event.x
        self.oldY = event.y

    def reset(self, event):
        self.oldX, self.oldY = None, None

    def classifyImage(self):
        size = 25, 25
        im = Image.open('input')
        im.thumbnail(size, Image.ANTIALIAS)
        
        im = im.convert('L')
        width, height = im.size
        pixels = im.load()
        #im.show()
        #invert, so that white is 0
        for y in range(width):
            for x in range(height):
                pixels[x, y] = 255 - im.getpixel((x,y))
        #Executing the nearest neighbor classifier (peter)
        
        input_data = {"label" : "", "image": im}
        NearestNeighbor.trim_whitespace([input_data])
        #(input_data["image"]).show()
        NearestNeighbor.to_numpy([input_data])
        NearestNeighbor.scale_down([input_data], 12, "average")
        n = NearestNeighbor.NearestNeighbor()
        result = n.classify_one(training_data, input_data)
        #Added pop up window that displays result and is destroyed when the user clicks 'clear'
        #print(result)
        message = Message(self.top, text = result)
        message.config(bg = 'white', font = ('arial', 60, 'bold'))
        message.pack()
        

if __name__ == '__main__':
    main()
