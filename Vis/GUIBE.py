import numpy as np
import math
import logging
from functools import partial
import Game as g
from Game import Position
import AI.DEC as dec
import AI.PCNN as pcnn

logging.basicConfig(level=logging.DEBUG)

def xyToFlat(x,y):
	return x*11+y

def flatToXY(x):
	return (math.floor(x/11), x%11)

g1 = g.Game()
# p1 = dec.DEC('AI_DEC')
p1 = pcnn.PCNN('pai1', pModel = 'Models/pcnn_p1_2_model.h5')

def moveClick(x):
	x,y = flatToXY(x)
	if g1.isMoveValid(Position(x,y)):
		g1.move(Position(x,y), player=1)
		tempM = p1.move(g1)
		g1.move(tempM)
		return(True,xyToFlat(tempM.x,tempM.y))
	else:
		return(False,0)