import numpy as np
import Game as g
from scipy import ndimage as sp
from AI.PAI import PAI
import math

bDim = (11,11)

strat1 = np.asarray([[0.1,0.1,0.1,0.1,0.3,0.3,0.7,0.9,0.9,0.7,0.3],
			[0.1,0.1,0.1,0.1,0.3,0.6,0.7,1,0.7,0.6,0.3],
			[0.1,0.1,0.1,0.3,0.3,0.7,0.9,0.9,0.7,0.3,0.3],
			[0.1,0.1,0.1,0.3,0.6,0.7,1,0.7,0.6,0.3,0.1],
			[0.1,0.1,0.3,0.3,0.7,0.9,0.9,0.7,0.3,0.3,0.1],
			[0.1,0.1,0.3,0.6,0.7,1,0.7,0.6,0.3,0.1,0.1],
			[0.1,0.3,0.3,0.7,0.9,0.9,0.7,0.3,0.3,0.1,0.1],
			[0.1,0.3,0.6,0.7,1,0.7,0.6,0.3,0.1,0.1,0.1],
			[0.3,0.3,0.7,0.9,0.9,0.7,0.3,0.3,0.1,0.1,0.1],
			[0.3,0.6,0.7,1,0.7,0.6,0.3,0.1,0.1,0.1,0.1],
			[0.3,0.7,0.9,0.9,0.7,0.3,0.3,0.1,0.1,0.1,0.1]])

strat2 = np.asarray([[0.1,0.3,0.6,0.7,0.9,1,0.9,0.7,0.6,0.3,0.1],
			[0.1,0.3,0.6,0.7,0.9,1,0.9,0.7,0.6,0.3,0.1],
			[0.1,0.3,0.6,0.7,0.9,1,0.9,0.7,0.6,0.3,0.1],
			[0.1,0.3,0.6,0.7,0.9,1,0.9,0.7,0.6,0.3,0.1],
			[0.1,0.3,0.6,0.7,0.9,1,0.9,0.7,0.6,0.3,0.1],
			[0.1,0.3,0.6,0.7,0.9,1,0.9,0.7,0.6,0.3,0.1],
			[0.1,0.3,0.6,0.7,0.9,1,0.9,0.7,0.6,0.3,0.1],
			[0.1,0.3,0.6,0.7,0.9,1,0.9,0.7,0.6,0.3,0.1],
			[0.1,0.3,0.6,0.7,0.9,1,0.9,0.7,0.6,0.3,0.1],
			[0.1,0.3,0.6,0.7,0.9,1,0.9,0.7,0.6,0.3,0.1],
			[0.1,0.3,0.6,0.7,0.9,1,0.9,0.7,0.6,0.3,0.1]])


strat3 = np.asarray([[0.9,1,0.9,0.7,0.6,0.3,0.3,0.1,0.1,0.1,0.1],
			[0.9,1,0.9,0.7,0.6,0.3,0.3,0.1,0.1,0.1,0.1],
			[0.9,1,0.9,0.7,0.6,0.3,0.3,0.1,0.1,0.1,0.1],
			[0.9,1,0.9,0.7,0.6,0.3,0.3,0.1,0.1,0.1,0.1],
			[0.9,1,0.9,0.7,0.6,0.3,0.3,0.1,0.1,0.1,0.1],
			[0.9,1,0.9,0.7,0.6,0.3,0.3,0.1,0.1,0.1,0.1],
			[0.9,1,0.9,0.7,0.6,0.3,0.3,0.1,0.1,0.1,0.1],
			[0.9,1,0.9,0.7,0.6,0.3,0.3,0.1,0.1,0.1,0.1],
			[0.9,1,0.9,0.7,0.6,0.3,0.3,0.1,0.1,0.1,0.1],
			[0.9,1,0.9,0.7,0.6,0.3,0.3,0.1,0.1,0.1,0.1],
			[0.9,1,0.9,0.7,0.6,0.3,0.3,0.1,0.1,0.1,0.1]])


strat4 = np.asarray([[0.1,0.1,0.1,0.1,0.3,0.3,0.6,0.7,0.9,1,0.9],
			[0.1,0.1,0.1,0.1,0.3,0.3,0.6,0.7,0.9,1,0.9],
			[0.1,0.1,0.1,0.1,0.3,0.3,0.6,0.7,0.9,1,0.9],
			[0.1,0.1,0.1,0.1,0.3,0.3,0.6,0.7,0.9,1,0.9],
			[0.1,0.1,0.1,0.1,0.3,0.3,0.6,0.7,0.9,1,0.9],
			[0.1,0.1,0.1,0.1,0.3,0.3,0.6,0.7,0.9,1,0.9],
			[0.1,0.1,0.1,0.1,0.3,0.3,0.6,0.7,0.9,1,0.9],
			[0.1,0.1,0.1,0.1,0.3,0.3,0.6,0.7,0.9,1,0.9],
			[0.1,0.1,0.1,0.1,0.3,0.3,0.6,0.7,0.9,1,0.9],
			[0.1,0.1,0.1,0.1,0.3,0.3,0.6,0.7,0.9,1,0.9],
			[0.1,0.1,0.1,0.1,0.3,0.3,0.6,0.7,0.9,1,0.9]])

strat_debug = np.ones(bDim)

# more strat_1 included as I personally think it is the better strategy.
strat = {
	0: strat1,
	1: strat2,
	2: strat3,
	3: strat4,
	4: strat1
}

### TODO elongation and	connection strategie
## The elongation strategy is already embedded in the general strategy

class DEC(PAI):
	def __init__(self, n, elo = 1000, nStrat = None, wRandMoves = 1):
		PAI.__init__(self, n, elo)
		self.nStrat = nStrat
		self.wRandMoves = wRandMoves
		if nStrat is None:
			self.strategie = strat[math.floor(np.random.rand(1)*len(strat))]
		else:
			self.strategie = strat[self.nStrat]
		self.moves = np.random.rand(*bDim)*self.wRandMoves
				
	def move(self, ga):
		if(np.random.rand(1)<0.2):
			self.moves = np.random.rand(*bDim)
## the ai tries to block the opponent's path. Redundant blocking is discouraged.
		temp1 = sp.correlate(ga.board.board, [[[-1,0],[0,1],[-1,-1]]], mode = 'constant', origin = (0,1,0))
		temp2 = sp.correlate(ga.board.board, [[[-1,-1],[0,1],[-1,0]]], mode = 'constant', origin = (0,-1,0))
		tempD = np.maximum(temp1[:,:,1],0) + np.maximum(temp2[:,:,1],0)
		temp = np.unravel_index(np.argmax(np.multiply(ga.validMoves(),self.moves+self.strategie+np.multiply(tempD, self.strategie))),bDim)
		return g.Position(*temp)
	
	def train(self, x=0, y=0,eps=0):
		if self.nStrat is None:
			self.strategie = strat[math.floor(np.random.rand(1)*len(strat))]
		else:
			self.strategie = strat[nStrat]
		self.moves = np.random.rand(*bDim)*self.wRandMoves
	
	def debug(self):
		self.moves = np.zeros(bDim)
		self.strategie = strat_debug

	


