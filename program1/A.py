import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D


import config
import load_data

def scatter_plot(arr):
	xmin, xmax = min(arr[:,0]), max(arr[:,0])
	ymin, ymax = min(arr[:,1]), max(arr[:,1])
	f, ax = plt.subplots(figsize=(7, 7))
	ax.scatter(arr[:,0], arr[:,1], marker='o', color='green', s=4, alpha=0.3)
	plt.title('title')
	plt.ylabel('latitude')
	plt.xlabel('longitude')
	#ftext = 'optional text'
	#plt.figtext(.15,.85, ftext, fontsize=11, ha='left')
	plt.ylim([ymin,ymax])
	plt.xlim([xmin,xmax])

	plt.show()	

def plot_3d(arr):
	xmin, xmax = min(arr[:,0]), max(arr[:,0])
	ymin, ymax = min(arr[:,1]), max(arr[:,1])
	fig = plt.figure(figsize=(10, 7))
	ax = fig.gca(projection='3d')
	x,y = arr[:,0], arr[:,1]
	nbins = 10
	Z,X,Y = np.histogram2d(x,y,bins=nbins)
	X = X[1:]
	Y = Y[1:]
	print Z,X,Y
	print len(Z), len(X), len(Y)
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
	        linewidth=0, antialiased=False)

	ax.set_zlim(0, 150)
	ax.zaxis.set_major_locator(plt.LinearLocator(10))
	ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('p(x)')

	plt.title('Bivariate Gaussian distribution')

	fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.coolwarm)

	plt.show()

def plot_hist(arr):
	xmin, xmax = min(arr[:,0]), max(arr[:,0])
	ymin, ymax = min(arr[:,1]), max(arr[:,1])	
	X,Y = arr[:,0], arr[:,1]

	fig1 = plt.figure()
	plt.plot(X,Y,'.r')
	plt.xlabel('x')
	plt.ylabel('y')

	nbins = 20
	H, xedges, yedges = np.histogram2d(X,Y,bins=nbins)
	H = np.rot90(H)
	H = np.flipud(H)
	Hmasked = np.ma.masked_where(H==0,H)

	fig2 = plt.figure()
	plt.pcolormesh(xedges,yedges,Hmasked)
	plt.xlabel('x')
	plt.ylabel('y')
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('Counts')

	plt.show()

def main():
	arr1,arr3 = load_data.A(config.TRAIN_DATA)
	#scatter_plot(arr1)
	#scatter_plot(arr3)
	plot_3d(arr1)
	#plot_hist(arr1)


if __name__ == '__main__':
    main()