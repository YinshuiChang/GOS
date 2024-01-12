import numpy as np
# from itertools import compress
import logging

class Position:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __str__(self):
		return 'p(%s, %s)'%(self.x, self.y)
		
	def __eq__(self, p):
		return self.x == p.x and self.y == p.y
		
	def isValid(self):
		return self.x>=0 and self.x<=10 and self.y>=0 and self.y<=10
	
	def copy(self):
		return Position(self.x, self.y)

bDim = (11,11)


class Board:
			
	def __init__(self):
			self.board = np.zeros((*bDim,2), dtype = np.int8)
		
	def __str__(self):
		return str(self.board[:,:,0]+self.board[:,:,1]*2)
	
	def __eq__(self, b):
		return np.sum(np.absolute(self.board - b.board)) == 0
		
	def setBoard(self, x):
		self.board = x.copy()

	def copy(self):
		output = Board()
		output.setBoard(self.board)
		return output
		
def boardArrayToStr(x):
	return str(x[:,:,0]+x[:,:,1]*2)
		

class Game:
	def __init__(self):
		self.board = Board()

	def __str__(self):
		return str(self.board)
	
	def validMoves(self):
		return np.maximum(np.ones(bDim)-self.board.board[:,:,0]-self.board.board[:,:,1],0)
		
	def isBoardFull(self):
		return sum(sum(self.validMoves())) == 0
	
	def flip(self):
		temp = np.empty(bDim, dtype = np.int8)
		temp = np.transpose(self.board.board[:,:,1]).copy()
		self.board.board[:,:,1] = np.transpose(self.board.board[:,:,0])
		self.board.board[:,:,0] = temp
	
	def move(self, p, player = 0):
		self.board.board[p.x,p.y,player] = 1

	def hasWon(self, player = 0):
		temp = self.initial()
		return any(map(isFinal, self.dfs(temp, temp)))
	
	def isP0At(self, p):
		return self.board.board[p.x,p.y,0] == 1
	
	def isMoveValid(self, p):
		return (self.board.board[p.x,p.y,0] + self.board.board[p.x,p.y,1]) == 0
	
	def dfs(self, xs, v):
		if not xs:
			return v
		x = xs.pop(0)
		temp = filter(self.isP0At, neighbor(x))
		temp = list(filter(lambda x: x not in v, temp))
		return self.dfs(temp + xs, v + temp)
			
	def initial(self):
		temp = np.multiply(self.board.board[0,:,0],range(1,12))-1
		temp = temp[temp >= 0]
		output = []
		for x in temp:
			output.append(Position(0,x))
		return output

		

def neighbor(p):
	temp = (Position(p.x-1,p.y),Position(p.x,p.y-1),Position(p.x+1,p.y),Position(p.x,p.y+1),Position(p.x+1,p.y-1),Position(p.x-1,p.y+1))
	return filter(Position.isValid, temp)

def isFinal(p):
	return p.x == 10


# dropout = [0.8,0.76,0.72,0.68,0.64,0.6,0.56,0.52,0.48,0.44,0.4,0.36,0.32,0.28,0.24,0.2,0.16,0.12,0.08,0.04]
# 	
# class BoardMoveDB:
# 	def __init__(self):
# 		self.boardDB = []
# 		self.moveDB = []
# 	
# 	def addBM(self, b, p):
# 		self.boardDB.append(b.board.copy())
# 		self.moveDB.append(p.x*11+p.y)
# 	
# 	def addDB(self, bm):
# ### Remove the first few moves from the DB randomly as they are mostly redundant
# 		fil = (np.random.rand(20)-dropout) > 0
# 		b = list(compress(bm.boardDB[:20], fil))
# 		m = list(compress(bm.moveDB[:20], fil))
# 		self.boardDB = self.boardDB + b + bm.boardDB[20:]
# 		self.moveDB = self.moveDB + m + bm.moveDB[20:]
# 		temp = len(self.boardDB) - 50000
# 		if temp > 0:
# 			del self.boardDB[:temp]
# 			del self.moveDB[:temp]

class TempBMDB:
	def __init__(self):
		self.bmdb = []
	
	def addBM(self, b, p):
		self.bmdb.append((b.board.copy(),p.x*11+p.y))
		

class BoardMoveDB:
	def __init__(self, dbSize = 50000, memory = 0.98, forget = 0.):
		self.dbSize = dbSize
		self.memory = memory
		self.forget = forget
		self.boardDB = []
		self.moveDB = []
		self.rewardDB = []
	
	def filterDB(self, bmdb, win):
		yield from []
		n = len(bmdb.bmdb)
		for b,m in bmdb.bmdb:
			if np.random.uniform() < (self.memory**n)*(1-self.forget)+self.forget:
				if win:
					yield(b,m,(100-n)/(1+len(bmdb.bmdb)/100))
				else:
					yield(b,m,(n-100)/(1+len(bmdb.bmdb)/100))
			n -= 1
	
	def addDB(self, bmdb, win=True):
		filBM = self.filterDB(bmdb, win)
		try:
			b, m, r = zip(*filBM)
		except:
			logging.debug('Generator is empty')
			return False
		self.boardDB = self.boardDB + list(b)
		self.moveDB = self.moveDB + list(m)
		self.rewardDB = self.rewardDB + list(r)
		temp = len(self.boardDB) - self.dbSize
		if temp > 0:
			del self.boardDB[:temp]
			del self.moveDB[:temp]
			del self.rewardDB[:temp]
		return True
			
class RandomAI:
	def __init__(self):
		self.moves = np.random.rand(*bDim)+1
	
	def move(self,g):
		temp = np.unravel_index(np.argmax(np.multiply(g.validMoves(),self.moves)),bDim)
		return Position(*temp)
