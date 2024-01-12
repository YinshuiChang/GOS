import numpy as np
import Game as g
import AI.PCNN as pcnn
import AI.DEC as dec
import math
from Vis.VisGoSBoard import plotBoard

bDim = g.bDim

def randomGame():
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
			d1.addBM(g1.board, tempM)
		else:
			tempM = p2.move(g1)
			d2.addBM(g1.board, tempM)
		g1.move(tempM)
		if g1.hasWon() or counter >= 121:
			break
		counter += 1
		g1.flip()

	if counter%2 == 0:
		d0.addDB(d1)
	else:
		d0.addDB(d2)


def playerAI(x):
	return player.get(x, 'invalid player')

def randPlayer():
	return player.get(math.floor(np.random.rand(1)*len(player)))


def aiGame():
	g1 = g.Game()
	d1 = g.TempBMDB()
	d2 = g.TempBMDB()
	counter = 0
	firstPl = randPlayer()
	secondPl = randPlayer()
	while True:
		if counter%2 == 0:
			tempM = firstPl.move(g1)
			d1.addBM(g1.board, tempM)
		else:
			tempM = secondPl.move(g1)
			d2.addBM(g1.board, tempM)
		g1.move(tempM)
		if g1.hasWon() or counter >= 121:
			break
		counter += 1
		g1.flip()
	if counter%2 == 0:
# 		print(firstPl, 'won; ', secondPl, 'lost')
		d0.addDB(d1)
	else:
# 		print(secondPl, 'won; ', firstPl, 'lost')
		d0.addDB(d2)

	
def trainingSession(eps = 1):
	for x in range(len(playerUniq)):
		playerUniq.get(x).train(np.asarray(d0.boardDB), np.asarray(d0.moveDB), eps)


def testAI(x,y):
	g1 = g.Game()
	counter = 0
	while True:
		if counter%2 == 0:
			tempM = x.move(g1)
		else:
			tempM = y.move(g1)
		g1.move(tempM)
		if g1.hasWon() or counter >= 121:
			break
		counter += 1
		g1.flip()

	if counter%2 == 0:
		print(x.n, ' has won')
	else:
		print(y.n, ' has won')
		g1.flip()
	plotBoard(g1.board)



if __name__ == '__main__':
	d0 = g.BoardMoveDB()

	for x in range(800):
		randomGame()

	pai1 = pcnn.PCNN('pai1')
	pai2 = pcnn.PCNN('pai2')
	pDEC1 = dec.DEC('pDEC1')
	pDEC2 = dec.DEC('pDEC2')
	pDEC3 = dec.DEC('pDEC3')
	pDEC4 = dec.DEC('pDEC4')
	pDEC5 = dec.DEC('pDEC5')
	pDEC6 = dec.DEC('pDEC6')
	# pai1.model.summary()



	pai1.train(np.asarray(d0.boardDB), np.asarray(d0.moveDB), 5)
	pai2.train(np.asarray(d0.boardDB), np.asarray(d0.moveDB), 5)

	# multiple AI player so the DEC strategies are more diverse while training
	player = {
			0: pai1,
			1: pai2,
			2: pDEC1,
			3: pDEC2,
			4: pai1,
			5: pai2,
			6: pDEC3,
			7: pDEC4,
			8: pai1,
			9: pai2,
			10: pDEC5,
			11: pDEC6
		}

	playerUniq = {
			0: pai1,
			1: pai2,
			2: pDEC1,
			3: pDEC2,
			4: pDEC3,
			5: pDEC4,
			6: pDEC5,
			7: pDEC6
		}
	
	nTrain = 10
	for y in range(nTrain):
		for x in range(100):
			aiGame()
		print('TS:' + str(y) +'/' + str(nTrain))
		trainingSession(2)

	for x in range(3):
		testAI(pai1,dec.DEC('p2',nStrat = 0, wRandMoves = 0.1))