__author__ = 'thomas'

import os
import matplotlib.pyplot as plt
try:
   import cPickle as pickle
except:
   import pickle

thresholdMin = 0.0
thresholdMax = 1.0
thresholdIncrement = 0.1

cosPickleFile = os.path.join(os.getcwd(), "cache/sims/sims.cos")
cosSimsList = pickle.load(open(cosPickleFile, "rb"))
hellPickleFile = os.path.join(os.getcwd(), "cache/sims/sims.hell")
hellSimsList = pickle.load(open(hellPickleFile, "rb"))

numNodes = 500
numEdgesFullyConnected = len(cosSimsList)


cosInfoArray = []
hellInfoArray = []
thresholdArray = [x * 0.01 for x in range(0, 101)]
for threshold in thresholdArray:
    print "calculating info for threshold = " + str(threshold)
    numCosEdges = 0
    numHellEdges = 0
    for fileid1, fileid2, cosSim in cosSimsList:
        if cosSim >= threshold:
            numCosEdges += 1
        else:
            break
    for fileid1a, fileid2a, hellSim in hellSimsList:
        if hellSim >= threshold:
            numHellEdges += 1
        else:
            break

    cosInfoArray.append(numCosEdges / float(numEdgesFullyConnected))
    hellInfoArray.append(numHellEdges / float(numEdgesFullyConnected))


# plt.plot(thresholdArray, cosInfoArray, 'r', thresholdArray, hellInfoArray, 'b')

cosInfoPlot, = plt.plot(thresholdArray, cosInfoArray, 'r', label='Cosine Similarity')
hellInfoPlot, = plt.plot(thresholdArray, hellInfoArray, 'b', label='Hellinger Similarity')
plt.legend(handles=[cosInfoPlot, hellInfoPlot])
plt.ylabel('numEdges / numEdgesFullyConnected')
plt.xlabel('similarity threshold')
plt.title('Connectivity')
fig = plt.gcf()
fig.canvas.set_window_title('Connectivity')
plt.show()
print "done"