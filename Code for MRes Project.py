'''Code from Markus Ferdinand Dablander
'''


import numpy as np
import hvapy_master.hva.HVA
import math
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats
from my_modules import entropy_estimators as ee

def hvg(ts): 
	'''Takes iterable ts containing a time series and produces the adjacency matrix of the resulting horizontal visibility network'''
	n = len(ts)
	A = np.zeros((n,n),int)
	for i in range(0,n-1):
		for j in range(i+1,n):
			if j==i+1:
				A[i,j]=1
			elif j>i+1:
				if np.amax(ts[i+1:j])<np.amin([ts[i],ts[j]]):
					A[i,j]=1	
			if ts[j]>=ts[i]:
				break
	return A+np.transpose(A)

def hvg_opt(ts):
	"""
    Function to calculate horizontal visiblity graph matrix from a
    1D-timeseries.
    Inputs:
    :param ts: 1xN Float32/double numpy array of the time series.

    Outputs:
    :return: NxN matrix of nodes and edges
    :rtype: NxN int8 numpy array
    """
	ts = np.float32(ts)
	ts = list(ts)
	ts = list(map(float,ts))
	ts = np.array(ts)
	#ts = np.float32(ts)


	return hvapy_master.hva.HVA.network_optim(ts)

def hvg_inverted(A):
    '''Takes numpy array A (adjacency matrix of horizontal visibility graph) and transform it to a 1-dimensional time series'''
    ts = np.zeros(len(A))
    for i in range(len(A)-1):
        A[i,i+1]=0
        A[i+1,i]=0
    for i in range(len(A)):
        for j in range(len(A)): 
            if A[i,j] == 1:
                for k in range(i+1,j):
                    ts[k]-=1
                A[i,j] = 0
                A[j,i] = 0
    return np.array(ts)


def boxplot(d1,d2, ylim = (0,1), title_x = "TITLE_X", title_y = "TITLE_Y", title_d1 = "data1", title_d2 = "data2", figsize = (3,6), colour_1 = "lightgreen", colour_2 = "red", fontsize_x = 10, fontsize_y = 10, fontsize_d = 8):
    
    randomDists = [title_d1, title_d2]

    data = [d1, d2]

    fig, ax1 = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Now fill the boxes with desired colors
    boxColors = [colour_1, colour_2]
    numBoxes = 2
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        # Alternate between Dark Khaki and Royal Blue
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        ax1.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
                 color='w', markeredgecolor='k')

    xtickNames = plt.setp(ax1, xticklabels=randomDists)
    plt.setp(xtickNames, rotation=45, fontsize=fontsize_d)
    
    plt.xlabel(title_x, fontsize = fontsize_x)
    plt.ylabel(title_y, fontsize = fontsize_y)
	
    plt.ylim(ylim)

    plt.show()

def network_analysis_1_signal(tsmatrixlist, structural_descriptor, signal_1, plot = False, ytitle = "Structural Descriptor"):
    
    n=len(tsmatrixlist)
    x = np.array([])
    y = np.array([])
    
    for i in range(n):
        x = np.append(x,tsmatrixlist[i][1])
        y = np.append(y,structural_descriptor(hvg(tsmatrixlist[i][0][:,signal_1])))
    
    ylow = np.array([])
    yhigh = np.array([])
    
    for i in range(n):
        l = tsmatrixlist[i][1]
        if l<=0.3:
            ylow = np.append(ylow,y[i])
        if l>0.3:
            yhigh = np.append(yhigh,y[i])
    
    if plot == True:
	    
	    plt.figure()
	    plt.scatter(x,y)
	    plt.show()
	    
	    plt.figure()
	    plt.boxplot(np.transpose(np.array([ylow,yhigh])),0)
	    plt.xticks([1, 2], [' \n Normal Outcome', '\n Poor Outcome'])
	    plt.ylabel(ytitle)
	    plt.show()


	    print("Cases: ",n)
	    print("\n")
	    print("Normality test ylow: ",scipy.stats.normaltest(ylow))
	    print("\n")
	    print("Normality test yhigh: ",scipy.stats.normaltest(yhigh))
	    print("\n")
	    print("Normality test y: ",scipy.stats.normaltest(y))
	    print("\n")
	    print("Kolmogorov-Smirnov (mild/severe same distribution): ",scipy.stats.ks_2samp(ylow,yhigh))
	    print("\n")
	    print("Welch (mild/severe same distribution): ",scipy.stats.ttest_ind(ylow,yhigh, equal_var = False))
	    print("\n")
	    print("Mild Group: ", ylow)
	    print("\n")
	    print("Severe Group: ",yhigh)
	    print("\n")
	    print("Mean of low-group: ",np.mean(ylow))
	    print("\n")
	    print("Mean of high-group: ",np.mean(yhigh))
	    print("\n")
	    print("Standard Deviation of low-group: ",np.std(ylow))
	    print("\n")
	    print("Standard Deviation of high-group: ",np.std(yhigh))
    
    return [y, ylow, yhigh]


def network_analysis_2_signals(tsmatrixlist, structural_descriptor, signal_1, signal_2, plot = False, ytitle = "Structural Descriptor"):
    
    n=len(tsmatrixlist)
    x = np.array([])
    y = np.array([])
    
    for i in range(n):
        x = np.append(x,tsmatrixlist[i][1])
        y = np.append(y,structural_descriptor(hvg(tsmatrixlist[i][0][:,signal_1]),hvg(tsmatrixlist[i][0][:,signal_2])))
    
    ylow = np.array([])
    yhigh = np.array([])
    
    for i in range(n):
        l = tsmatrixlist[i][1]
        if l<=0.3:
            ylow = np.append(ylow,y[i])
        if l>0.3:
            yhigh = np.append(yhigh,y[i])
    
    if plot == True:
	    
	    plt.figure()
	    plt.scatter(x,y)
	    plt.show()
	    
	    plt.figure()
	    plt.boxplot(np.transpose(np.array([ylow,yhigh])),0)
	    plt.xticks([1, 2], [' \n Normal Outcome', '\n Poor Outcome'])
	    plt.ylabel(ytitle)
	    plt.show()


	    print("Cases: ",n)
	    print("\n")
	    print("Normality test ylow: ",scipy.stats.normaltest(ylow))
	    print("\n")
	    print("Normality test yhigh: ",scipy.stats.normaltest(yhigh))
	    print("\n")
	    print("Normality test y: ",scipy.stats.normaltest(y))
	    print("\n")
	    print("Kolmogorov-Smirnov (mild/severe same distribution): ",scipy.stats.ks_2samp(ylow,yhigh))
	    print("\n")
	    print("Welch (mild/severe same distribution): ",scipy.stats.ttest_ind(ylow,yhigh, equal_var = False))
	    print("\n")
	    print("Mild Group: ", ylow)
	    print("\n")
	    print("Severe Group: ",yhigh)
	    print("\n")
	    print("Mean of low-group: ",np.mean(ylow))
	    print("\n")
	    print("Mean of high-group: ",np.mean(yhigh))
	    print("\n")
	    print("Standard Deviation of low-group: ",np.std(ylow))
	    print("\n")
	    print("Standard Deviation of high-group: ",np.std(yhigh))
    
    return [y, ylow, yhigh]

def aeo_list(matrixlist): 
	''' takes a list of adjacency matrices of horizontal visibility networks and computes the average edge overlap of the corresponding multilayer network'''
	m = len(matrixlist)
	n = len(matrixlist[0])
	num = 0
	denom = 0
	for i in range(n):
		for j in range(i+1,n):
			alphasum = 0
			for alpha in range(m):
				alphasum += matrixlist[alpha][i,j]
			num += alphasum
			if alphasum == 0:
				delta = 1
			else:
				delta = 0
			denom += 1-delta
	denom *= m
	return num/denom


def aeo(*matrixlist):
	return aeo_list(matrixlist)

def mid(A1, A2):
	''' takes the adjacency matrices of two networks and computes the mutual information of the degree distributions of those two networks'''
	n = np.shape(A1)[1]
	I=0
	maxdeg1=np.amax(np.sum(A1, axis=1))
	maxdeg2=np.amax(np.sum(A2, axis=1))
	for k1 in range(maxdeg1+1):
		for k2 in range(maxdeg2+1):
			nk1=0
			nk2=0
			nk1k2=0
			rowsumA1 = np.sum(A1,axis=1)
			rowsumA2 = np.sum(A2,axis=1)
			for i in range(n):
				if rowsumA1[i] == k1:
					nk1+=1
				if rowsumA2[i] == k2:
					nk2+=1
				if rowsumA1[i] == k1 and rowsumA2[i]==k2:
					nk1k2+=1
			if nk1!=0 and nk2!=0 and nk1k2!=0:
				Pk1=nk1/n
				Pk2=nk2/n
				Pk1k2=nk1k2/n
				I += Pk1k2*math.log(Pk1k2/(Pk1*Pk2))
	return I

def ed(A1, norm = True):
	''' takes the adjacency matrix (numpy array) of a network and computes the entropy of the degree distribution. Needed for the normalization of the mutual information of degree distribution.'''
	entropy=0
	n = np.shape(A1)[1]
	maxdeg1 = np.amax(np.sum(A1, axis=1))
	for k1 in range(maxdeg1+1):
		nk1=0
		for i in np.sum(A1,axis=1):
			if i == k1:
				nk1+=1
		if nk1!=0:
			Pk1=nk1/n
			entropy += -Pk1*math.log(Pk1)

	if norm == True:
		entropy = entropy/(math.log(A1.shape[0]-1,2))


	return entropy

def mid_norm(A1, A2):
	'''Normalized Version of the mutual information of degree distribution. Takes values between 0 (Independence) and 1 (One degree distribution is fully determined by the other one, perfect dependence).'''
	mi = mid(A1,A2)
	return mi/(ed(A1)+ed(A2)-mi)

def mid_norm_ee(A1, A2, norm = True):
	'''Normalized Version of the mutual information of degree distribution. Takes values between 0 (Independence) and 1 (One degree distribution is fully determined by the other one, perfect dependence).'''
	
	deglist_1 = list(np.sum(A1, axis=1))
	deglist_2 = list(np.sum(A2, axis=1))
	
	mi = ee.midd(deglist_1,deglist_2)

	if norm == True:
		mi = mi/(ee.entropyd(list(zip(deglist_1,deglist_2))))

	return mi

def ted(A1, A2, tau = 1):
    '''Transfer Entropy of Degree Distribution with Time Lag 1 and horizontal visibility adjacency matrices A1,A2'''
    deg_seq_A1 = list(np.sum(A1, axis = 0))
    deg_seq_A2 = list(np.sum(A2, axis = 0))
    return ee.cmidd(deg_seq_A1[:-tau], deg_seq_A2[tau:], deg_seq_A2[:-tau])

def ted_norm(A1, A2, tau = 1):
    '''Transfer Entropy of Degree Distribution with Time Lag 1 and horizontal visibility adjacency matrices A1 -> A2'''
    deg_seq_A1 = list(np.sum(A1, axis = 0))
    deg_seq_A2 = list(np.sum(A2, axis = 0))
    return ee.cmidd(deg_seq_A1[:-tau], deg_seq_A2[tau:], deg_seq_A2[:-tau])/ee.cmidd(deg_seq_A2[tau:],deg_seq_A2[tau:],deg_seq_A2[:-tau])


def clc(A1):
	'''Average Clustering Coefficient of Graph associated with adjacency matrix A1'''
	G = nx.from_numpy_matrix(A1)
	return nx.average_clustering(G)

def matrix_to_gml_file(Gw, L, name):
    
    Gw = np.array(Gw)
    n = len(Gw)
    
    G = nx.from_numpy_matrix(Gw)
    
    ll = {}
    for k in list(range(n)):
        ll[k] = str(k) +": " + L[k]
    G = nx.relabel_nodes(G, ll)
    
    nx.write_gml(G, name + ".gml")


def plot_signals(signal_list, name = "Signals", linewidth = 1, ylim = (-3,3), xlim = (0,33)):
    
    l = len(signal_list[0])
    x = np.arange(0, l, 1)
    plt.figure(figsize=(20,4))
    
    for signal in signal_list:   
        y = np.array(list(signal)) 
        plt.plot(x,y, linewidth = linewidth)
        
    
    plt.xlabel("Time [s]", fontsize = 20)
    plt.ylabel(name, fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.show()
        

def plot_equivalence_classes(signal_length, signal_number, linewidth):
	'''
	Plots several signals with the same horizontal visibility network in order to visualize the information loss involved in the network transformation.
	'''

    x = int((signal_length-1)*(signal_length-2)/2)
        
    s = []
    for k1 in range(signal_length-2):
        for k2 in range(2+k1,signal_length):
            s.append((k1,k2))
    
    adj_matrices = []
    for i in range(1 << x):
        set = [s[j] for j in range(x) if (i & (1 << j))]
        A = np.zeros((signal_length, signal_length))
        
        for ind in set:
            A[ind[0],ind[1]] = 1
        
        
        A = A + np.transpose(A)
        
        for k1 in range(signal_length):
            for k2 in range(signal_length):
                if abs(k1-k2) == 1:
                    A[k1,k2]=1
        
        adj_matrices.append([A,[]])
        
    signals = np.random.randn(signal_number, signal_length)
    for sig in signals:
        hvg_sig = hvg(sig)
        for matrix in adj_matrices:
            if np.array_equal(hvg_sig, matrix[0]):
                matrix[1].append(list(sig))
    
    for matrix in adj_matrices:
        if len(matrix[1])>0:
            print("Adjacency Matrix of Horizontal Visibility Graph: \n \n ",matrix[0],"\n")
            print("Number of Elemens in Equivalence Class: ", len(matrix[1]),"\n")
            plot_signals(np.array(matrix[1]), linewidth = linewidth)
            print("\n \n \n \n \n") 


