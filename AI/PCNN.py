from tensorflow import keras
import numpy as np
from Game import Position
from AI.PAI import PAI
import logging

bDim = (11,11)

## Parallel Convolutional Neuronal Network
class PCNN(PAI):
	def __init__(self, n, elo = 1000, pModel = None):
		PAI.__init__(self, n, elo)
		
		if pModel is None:
			self.model_pred, self.model = self.createModel()
		else:
			self.model_pred = self.loadModel(pModel)
			output = keras.layers.Activation('softmax')(self.model_pred.output)
			self.model = keras.models.Model(self.model_pred.input, output)
			self.model.compile(optimizer='adam',
       						loss='sparse_categorical_crossentropy',
        					metrics=['accuracy'])
			logging.info(self.model.summary())
		


	def train(self, x, y, eps = 1):
		self.model.fit(x, y, epochs = eps)
	
	def evaluate(self, x, y):
		test_loss, test_acc = self.model.evaluate(x, y)
		print('Test accuracy:', test_acc)
		
	def move(self, g):
		moves = self.model_pred.predict(g.board.board.reshape(1,11,11,2))
		# Gumbel max -> sampling from the distribution -> more exploration
		moves = moves - np.log(-np.log(np.random.uniform(1e-5,1-1e-5,121)))
		temp = np.unravel_index(np.argmax(np.multiply(g.validMoves(),moves.reshape(bDim))),bDim)
		temp = Position(*temp)
		return temp
## greedy algorithm, alway takes the best move
# 	def move(self, g):
# 		moves = self.model.predict(g.board.board.reshape(1,11,11,2))
# 		temp = np.unravel_index(np.argmax(np.multiply(g.validMoves(),moves.reshape(bDim))),bDim)
# 		temp = Position(*temp)
# 		return temp
	
	def createModel(self):
		## The model is has four separate branches
		## 1) A 3x3 Convolution Deconvolution to learn local patterns
		## 2) A 4x4 Convolution Deconvolution
		## 3) A Pooling Dense UpSampling Network to learn global strategies
		## 4) a direct pass of the inputs to ease the determination of valid moves
		## the four branches are merged and a 1x1 Convolution determines the weights for the merged layers.
		inputs = keras.layers.Input(shape=(*bDim,2))
		
		branch_a = keras.layers.Conv2D(16, (3, 3), padding='valid')(inputs)
		## a leaky relu is used to counteract the vanishing gradient as the inputs are only zeros and ones
		branch_a = keras.layers.LeakyReLU(alpha=0.2)(branch_a)
		branch_a = keras.layers.Dropout(0.2)(branch_a)
		branch_a = keras.layers.Conv2DTranspose(4, (3,3), padding="valid", 
												activation=keras.activations.relu)(branch_a)
		
		branch_b = keras.layers.Conv2D(16, (4, 4), padding='valid')(inputs)
		branch_b = keras.layers.LeakyReLU(alpha=0.2)(branch_b)
		branch_b = keras.layers.Dropout(0.2)(branch_b)
		branch_b = keras.layers.Conv2DTranspose(4, (4,4), padding="valid", 
												activation=keras.activations.relu)(branch_b)
		
		branch_c = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='same')(inputs)
		branch_c = keras.layers.Flatten()(branch_c)
		branch_c = keras.layers.Dense(72, activation=keras.activations.relu,
									  kernel_regularizer=keras.regularizers.l2(0.05))(branch_c)
		branch_c = keras.layers.Reshape((6,6,2))(branch_c)
		branch_c = keras.layers.UpSampling2D(size=(2, 2))(branch_c)
		branch_c = keras.layers.Cropping2D(cropping=((0, 1), (0, 1)))(branch_c)
		
		merged = keras.layers.concatenate([inputs,branch_a,branch_b,branch_c], axis = -1)
		merged = keras.layers.Conv2D(filters=1, kernel_size=1, padding="valid")(merged)
		merged = keras.layers.Flatten()(merged)
		
		output = keras.layers.Activation('softmax')(merged)
		
		model_pred = keras.models.Model(inputs, merged)
		model = keras.models.Model(inputs, output)
		model.compile(optimizer='adam',
       						loss='sparse_categorical_crossentropy',
        					metrics=['accuracy'])
		return (model_pred, model)
	
	def saveModel(self, x):
		self.model_pred.save(x)
		
	def loadModel(self, x):
		return keras.models.load_model(x)

def main():
	import Game as g
	
	g1 = g.Game()
	p1 = PCNN('p1')
	d1 = g.TempBMDB()
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
	
	for x,y in d1.bmdb:
		print(g.boardArrayToStr(x))
		print(y)
		
	d0 = g.BoardMoveDB()
	d0.addDB(d1)

	
	p1.train(np.asarray(d0.boardDB), np.asarray(d0.moveDB), 10)
	
	g1 = g.Game()
	tempM = Position(2,4)
	g1.move(tempM)
	g1.flip()
	print(p1.model.predict(g1.board.board.reshape(1,11,11,2)))
	
	

	
	g1 = g.Game()
	print(g1.board)
	g1.move(p1.move(g1))
	print(g1.board)
	
	g1 = g.Game()
	tempM = Position(2,4)
	g1.move(tempM)
	g1.flip()
	print(g1.board)
	g1.move(p1.move(g1))
	print(g1.board)
	
	g1 = g.Game()
	tempM = Position(2,4)
	g1.move(tempM)
	g1.flip()
	tempM = Position(4,4)
	g1.move(tempM)
	g1.flip()
	print(g1.board)
	g1.move(p1.move(g1))
	print(g1.board)
	
	

if __name__ == '__main__':
	main()
	
	
	



