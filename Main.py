import numpy as np
import Game as g
import AI.PCNN as pcnn
import AI.DEC as dec
import Training as t
from Vis.VisGoSBoard import plotBoard

bDim = g.bDim


if __name__ == '__main__':
	t.d0 = g.BoardMoveDB()

	for x in range(1000):
		t.randomGame()

	pai1 = pcnn.PCNN('pai1')
	pai2 = pcnn.PCNN('pai2')
	pDEC1 = dec.DEC('pDEC1')
	pDEC2 = dec.DEC('pDEC2')
	pDEC3 = dec.DEC('pDEC3')
	pDEC4 = dec.DEC('pDEC4')
	pDEC5 = dec.DEC('pDEC5')
	pDEC6 = dec.DEC('pDEC6')
	# pai1.model.summary()



	pai1.train(np.asarray(t.d0.boardDB), np.asarray(t.d0.moveDB), 5)
	pai2.train(np.asarray(t.d0.boardDB), np.asarray(t.d0.moveDB), 5)

	# multiple AI player so the DEC strategies are more diverse while training
	t.player = {
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

	t.playerUniq = {
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
		for x in range(500):
			t.aiGame()
		print('TS:' + str(y) +'/' + str(nTrain))
		t.trainingSession(2)

	# pai1.saveModel('Models/pcnn_p1_model_x.h5')
	# pai2.saveModel('Models/pcnn_p2_model_x.h5')

	for x in range(3):
		t.testAI(pai1,dec.DEC('p2',nStrat = 0, wRandMoves = 0.1))