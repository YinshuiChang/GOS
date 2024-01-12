import numpy as np
import Game as g
import AI.DEC as dec
import logging
import math

def addRandomGames():
	g1 = g.Game()
	p1 = dec.DEC('p1')
	p2 = dec.DEC('p1')
	d1 = g.TempBMDB()
	d2 = g.TempBMDB()
	p1.train()
	p2.train()
	counter = 0
	while True:
		if counter%2 == 0:
			tempM = p1.move(g1)
			d1.addBM(g1.board, g.Position(math.floor(counter/11), counter%11))
		else:
			tempM = p2.move(g1)
			d2.addBM(g1.board, g.Position(math.floor(counter/11), counter%11))
		g1.move(tempM)
		if g1.hasWon() or counter >= 121:
			break
		counter += 1
		g1.flip()
	
	print(len(d1.bmdb)+len(d2.bmdb)-1)
	if counter%2 == 0:
		d0.addDB(bmdb = d1)
	else:
		d0.addDB(bmdb = d2)
		
logging.basicConfig(level=logging.DEBUG)
d0 = g.BoardMoveDB()
for i in range(10):
	addRandomGames()
print(d0.moveDB)