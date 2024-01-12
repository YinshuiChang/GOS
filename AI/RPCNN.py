from tensorflow import keras
import numpy as np
from Game import Position
from AI.PAI import PAI
import logging

bDim = (11,11)

## Parallel Convolutional Neuronal Network
class RPCNN(PAI):
	def __init__(self, n, elo = 1000, pModel = None):
		PAI.__init__(self, n, elo)
		
		self.model_train, self.model_pred = self.createModel()
		if pModel is None:
			self.model_train, self.model_pred = self.createModel()
		else:
			self.model_pred = self.loadModel(pModel)
			output = keras.layers.Activation('softmax')(self.model_pred.output)
			self.model = keras.models.Model(self.model_pred.input, output)
			self.model.compile(optimizer='adam',
       						loss='sparse_categorical_crossentropy',
        					metrics=['accuracy'])
			logging.info(self.model.summary())
		


	def train(self, xs, ys, zs, eps = 1):
		splitedDS = [[] for x in range(121)]
		for x, y, z in zip(xs,ys,zs):
			splitedDS[y].append((x,z))
		i = 0
		print(splitedDS)
		for y in splitedDS:
			try:
				x, z = zip(*y)
			except:
				logging.debug('no234566')
			print(x)
			print(z)
			self.model_train[i].fit(list(x), z, epochs = eps)
			i += 1
	
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
		branch_c = keras.layers.Dense(72, activation=keras.activations.relu)(branch_c)
		branch_c = keras.layers.Reshape((6,6,2))(branch_c)
		branch_c = keras.layers.UpSampling2D(size=(2, 2))(branch_c)
		branch_c = keras.layers.Cropping2D(cropping=((0, 1), (0, 1)))(branch_c)
		
		merged = keras.layers.concatenate([inputs,branch_a,branch_b,branch_c], axis = -1)
		merged = keras.layers.Conv2D(filters=3, kernel_size=1, padding="valid")(merged)
		merged = keras.layers.Flatten()(merged)
		branch_a = keras.layers.Dropout(0.3)(branch_a)
		merged = keras.layers.Dense(121, activation=keras.activations.relu)(merged)
		
		model_core = keras.models.Model(inputs, merged)
		
		pred = []
		model_train = []
		for i in range(121):
			temp_layer = keras.layers.Dense(1)(merged)
			temp_model = keras.models.Model(inputs, temp_layer)
			temp_model.compile(optimizer='adam',
       						loss='mean_absolute_error',
        					metrics=['accuracy'])
			pred.append(temp_layer)
			model_train.append(temp_model)
		print(pred)
		print(model_train[0].summary())
		output = keras.layers.concatenate(pred, axis = -1)
		
		model_pred = keras.models.Model(inputs, output)

		return (model_train, model_pred)
	
	def saveModel(self, x):
		self.model_pred.save(x)
		
	def loadModel(self, x):
		return keras.models.load_model(x)

def main():
	import Game as g
	import AI.DEC as dec
	dw = g.BoardMoveDB()
	dl = g.BoardMoveDB()
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
		dw.addDB(d1)
		dl.addDB(d2, win=False)
	else:
		dw.addDB(d2)
		dl.addDB(d1, win=False)
	
	
	pai = RPCNN('pai')
	pai.train(dw.boardDB, dw.moveDB, dw.rewardDB,5)
	

if __name__ == '__main__':
	main()
	
	
	



