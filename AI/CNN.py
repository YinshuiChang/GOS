import tensorflow as tf
import numpy as np
from Game import Position
from AI.PAI import PAI

bDim = (11,11)


class CNN(PAI):
	def __init__(self, n, elo = 1000):
		PAI.__init__(self, n, elo)
		self.model = tf.keras.models.Sequential([
  			tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(*bDim,2)),
  			tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  			tf.keras.layers.Flatten(),
  			tf.keras.layers.Dropout(0.2),
  			tf.keras.layers.Dense(242, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  			tf.keras.layers.Dropout(0.2),
  			tf.keras.layers.Dense(units=11*11, activation=tf.nn.softmax),
		])
		self.model.compile(optimizer='adam',
       		loss='sparse_categorical_crossentropy',
        	metrics=['accuracy'])

	def train(self, x, y, eps = 1):
		self.model.fit(x, y, epochs = eps)
	
	def evaluate(self, x, y):
		test_loss, test_acc = self.model.evaluate(x, y)
		print('Test accuracy:', test_acc)
		
	def move(self, g):
		moves = self.model.predict(g.board.board.reshape(1,11,11,2))
		temp = np.unravel_index(np.argmax(np.multiply(g.validMoves(),moves.reshape(bDim))),bDim)
		temp = Position(*temp)
		return temp
	
	def saveModel(self, x):
		self.model.save(x)
	

def main():
	import Game as g
	
	g1 = g.Game()
	p1 = CNN('p1')
	d1 = g.BoardMoveDB()
	tempM = Position(2,4)
	d1.addBM(g1.board, tempM)
	g1.move(tempM)
	g1.flip()
	tempM = Position(4,4)
	d1.addBM(g1.board, tempM)
	g1.move(tempM)
	g1.flip()
	tempM = Position(5,5)
	d1.addBM(g1.board, tempM)
	g1.move(tempM)
	
	for x in d1.boardDB:
		print(g.boardArrayToStr(x))
	
	print(d1.moveDB)
	
	p1.train(np.asarray(d1.boardDB), np.asarray(d1.moveDB), 100)
	
	g1 = g.Game()
	tempM = Position(2,4)
	g1.move(tempM)
	g1.flip()
	print(p1.model.predict(g1.board.board.reshape(1,11,11,2)))
	
	g1 = g.Game()
	g1.move(p1.move(g1))
	print(g1.board)
	g1.flip()
	g1.move(p1.move(g1))
	print(g1.board)
	g1.flip()
	g1.move(p1.move(g1))
	print(g1.board)
	g1.flip()

if __name__ == '__main__':
	main()
	
	
	



