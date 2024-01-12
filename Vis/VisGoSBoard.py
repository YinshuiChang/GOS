# imports
import matplotlib.pyplot as plt
from matplotlib.collections import RegularPolyCollection
import numpy as np
import Game as g

def plotBoard(b):
	fig, ax = plt.subplots(1)
	
	pixel_color = np.reshape(b.board, (121,2))
	pixel_color=np.concatenate((pixel_color, np.zeros(121).reshape(121,1)), axis = 1)
	
	temp_1 = []
	for i in range(11):
		temp_1=np.concatenate((temp_1, range(11)+np.ones(11)*0.5*i))
	temp_2 = np.repeat(range(11), 11)*0.65
	offsets = np.concatenate((temp_1,temp_2), axis=0)
	offsets = np.transpose(np.reshape(offsets,(2,121)))
	
	black = (0,0,0,1)

	collection = RegularPolyCollection(
	    numsides=6,
	    rotation=0,
	    sizes=(400,),
	    facecolors=pixel_color,
	    edgecolors=(black,),
	    linewidths=(1,),
	    offsets=offsets,
	    transOffset=ax.transData,
    )

	ax.add_collection(collection)
	ax.axis([-1, 16, -1, 8])
	plt.show()


if __name__ == '__main__':
	g1 = g.Game()
	tempM = g.Position(0,1)
	g1.move(tempM)
	g1.flip()
	tempM = g.Position(4,4)
	g1.move(tempM)
	g1.flip()
	tempM = g.Position(5,5)
	g1.move(tempM)
	print(g1.board)
	plotBoard(g1.board)


