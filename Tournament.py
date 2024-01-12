import numpy as np
import Game as g
import AI.DEC as dec
import AI.PCNN as pcnn
import Training as t
import logging

bDim = g.bDim


## function ist not working as it should. The AIs could not be set globally fron within the function
def loadAIs():
	pcnn1 = pcnn.PCNN('pcnn_p1', pModel = 'Models/pcnn_p1_model.h5')
	pcnn2 = pcnn.PCNN('pcnn_p1', pModel = 'Models/pcnn_p1_model.h5')
	pcnn3 = pcnn.PCNN('pcnn_p1', pModel = 'Models/pcnn_p1_model.h5')
	pcnn4 = pcnn.PCNN('pcnn_p2', pModel = 'Models/pcnn_p2_model.h5')
	pcnn5 = pcnn.PCNN('pcnn_p2', pModel = 'Models/pcnn_p2_model.h5')
	pcnn6 = pcnn.PCNN('pcnn_p2', pModel = 'Models/pcnn_p2_model.h5')
# 	cnn1 = cnn.CNN('cnn_p1', pModel = 'Models/cnn_p1_model.h5')
# 	cnn2 = cnn.CNN('cnn_p1', pModel = 'Models/cnn_p1_model.h5')
# 	cnn3 = cnn.CNN('cnn_p2', pModel = 'Models/cnn_p2_model.h5')
# 	cnn4 = cnn.CNN('cnn_p2', pModel = 'Models/cnn_p2_model.h5')
	dec1 = dec.DEC('DEC')
	dec2 = dec.DEC('DEC')
	dec3 = dec.DEC('DEC')



def getElo():
	for x in playerL:
		yield x.elo

def testLAI():
	for _ in range(3):
		t.testAI(pai1,dec.DEC('p2',nStrat = 0, wRandMoves = 0.1))
	for _ in range(3):
		t.testAI(pai2,dec.DEC('p2',nStrat = 0, wRandMoves = 0.1))

def tGame():
	g1 = g.Game()
	d1 = g.TempBMDB()
	d2 = g.TempBMDB()
	counter = 0
	firstPl, secondPl = randPlayerPair()
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
		d0.addDB(d1)
		firstPl.elo, secondPl.elo = eloUpdate(firstPl.elo, secondPl.elo, True)
	else:
		d0.addDB(d2)
		firstPl.elo, secondPl.elo = eloUpdate(firstPl.elo, secondPl.elo, False)

def playerAI(x):
	return player.get(x, 'invalid player')

def randPlayerPair():
	pList = list(map(lambda x: x/(nPLayer*1000), getElo()))
	logging.debug(pList)
	logging.debug(sum(pList))
	rPlayer = np.random.choice(nPLayer, 2, replace=False, p = pList)
	return (playerD.get(rPlayer[0]),playerD.get(rPlayer[1]))

def eloUpdate(p1,p2,p1Won):
	pred = 1 / (1 + 10**((p2-p1)/400))
	if p1Won:
		temp = 32*(1-pred)
	else:
		temp = 32*(-pred)
	return(p1+temp, p2-temp)

def trainingSession(eps = 1):
	for x in playerL:
		x.train(np.asarray(d0.boardDB), np.asarray(d0.moveDB), eps)

def main_1():
	logging.basicConfig(level=logging.INFO)
	pcnn1 = pcnn.PCNN('pcnn_p1', pModel = 'Models/pcnn_p1_model.h5')
	pcnn2 = pcnn.PCNN('pcnn_p1', pModel = 'Models/pcnn_p1_model.h5')
	pcnn3 = pcnn.PCNN('pcnn_p1', pModel = 'Models/pcnn_p1_model.h5')
	pcnn4 = pcnn.PCNN('pcnn_p2', pModel = 'Models/pcnn_p2_model.h5')
	pcnn5 = pcnn.PCNN('pcnn_p2', pModel = 'Models/pcnn_p2_model.h5')
	pcnn6 = pcnn.PCNN('pcnn_p2', pModel = 'Models/pcnn_p2_model.h5')
	dec1 = dec.DEC('DEC')
	dec2 = dec.DEC('DEC')
	dec3 = dec.DEC('DEC')
	nPLayer = 9

	playerD = {
		0: pcnn1,
		1: pcnn2,
		2: pcnn3,
		3: pcnn4,
		4: pcnn5,
		5: pcnn6,
		6: dec1,
		7: dec2,
		8: dec3
	}

	playerL = [pcnn1, pcnn2, pcnn3, pcnn4, pcnn5, pcnn6, dec1, dec2, dec3]
	
	d0 = g.BoardMoveDB()
	
	nTrain = 10
	for y in range(nTrain):
		for x in range(500):
			tGame()
		print('TS:' + str(y) +'/' + str(nTrain))
		trainingSession(2)
	

def main_2():
	print(eloUpdate(1999.999,0.001,True))

if __name__ == '__main__':
	logging.basicConfig(level=logging.WARNING)
	pcnn1 = pcnn.PCNN('pcnn_p1', pModel = 'Models/pcnn_p1_model.h5')
	pcnn2 = pcnn.PCNN('pcnn_p1', pModel = 'Models/pcnn_p1_model.h5')
	pcnn3 = pcnn.PCNN('pcnn_p1', pModel = 'Models/pcnn_p1_model.h5')
	pcnn4 = pcnn.PCNN('pcnn_p2', pModel = 'Models/pcnn_p2_model.h5')
	pcnn5 = pcnn.PCNN('pcnn_p2', pModel = 'Models/pcnn_p2_model.h5')
	pcnn6 = pcnn.PCNN('pcnn_p2', pModel = 'Models/pcnn_p2_model.h5')
	dec1 = dec.DEC('DEC')
	dec2 = dec.DEC('DEC')
	dec3 = dec.DEC('DEC')
	nPLayer = 9

	playerD = {
		0: pcnn1,
		1: pcnn2,
		2: pcnn3,
		3: pcnn4,
		4: pcnn5,
		5: pcnn6,
		6: dec1,
		7: dec2,
		8: dec3
	}

	playerL = [pcnn1, pcnn2, pcnn3, pcnn4, pcnn5, pcnn6, dec1, dec2, dec3]
	
	d0 = g.BoardMoveDB()
	
	nTrain = 100
	for y in range(nTrain):
		for x in range(500):
			tGame()
		print('TS:' + str(y) +'/' + str(nTrain))
		trainingSession(2)
	
	pcnn1.saveModel('Models/pcnn_p1_1_model.h5')
	pcnn2.saveModel('Models/pcnn_p1_2_model.h5')
	pcnn3.saveModel('Models/pcnn_p1_3_model.h5')
	pcnn4.saveModel('Models/pcnn_p2_4_model.h5')
	pcnn5.saveModel('Models/pcnn_p2_5_model.h5')
	pcnn6.saveModel('Models/pcnn_p2_6_model.h5')
	