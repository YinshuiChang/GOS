from tkinter import *
import numpy as np
import logging
from Vis.GUIBE import *
from functools import partial

logging.basicConfig(level=logging.INFO)

size = 2

canvas_width = 350 * size
canvas_height = 250 * size

def Hexagon(canvas, x,y,z, outline="#476042", fill='yellow', width = 1):
   points = [17.32,5,8.66,0,0,5,0,15,8.66,20,17.32,15]
   shift = [x,y,x,y,x,y,x,y,x,y,x,y]
   comb = [i*size+j for i,j in zip(points,shift)]
   return canvas.create_polygon(comb, outline=outline, 
                         fill=fill, width=width, 
                         tags="hex"+ str(z))

def callback(event, nHex = -1):
	boolM, nMove = moveClick(nHex)
	if boolM:
		w.itemconfig(hexas[nHex], fill="blue")
		w.itemconfig(hexas[nMove], fill="red")
	else:
		print('invalidMove')
	logging.info('Clicked at x:'+ str(event.x) +', y:' + str(event.y) +' on Hex' + str(nHex))

master = Tk()

w = Canvas(master, 
           width=canvas_width, 
           height=canvas_height)

temp_1 = []
for i in range(11):
	temp_1=np.concatenate((temp_1, range(11)+np.ones(11)*0.5*i))
temp_1 *= 20*size
temp_2 = np.repeat(range(10,-1,-1), 11)*18*size
temp_1 += 10
temp_2 += 10
offsets = zip(temp_1,temp_2)

k = 0
hexas = []
for i,j in offsets:
	hexas.append(Hexagon(w,i,j,k,outline='red',fill='gold', width=1))
	w.tag_bind('hex'+ str(k), "<Button-1>", partial(callback, nHex = k))
	k += 1

w.pack()

mainloop()
