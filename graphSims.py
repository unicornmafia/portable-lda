__author__ = 'thomas'

import os
import networkx as nx
import matplotlib.pyplot as plt
try:
   import cPickle as pickle
except:
   import pickle


pickleFile = os.path.join(os.getcwd(), "cache/sims/sims.hell")
simsDict = pickle.load(open(pickleFile, "rb"))

G = nx.Graph()

weights = []
for fileid1 in simsDict.keys():
    for filesim in simsDict[fileid1]:
        fileid2 = filesim[0]
        sim = filesim[1]
        G.add_edge(fileid1, fileid2, weight=sim)

e1 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.2]
e2 = [(u, v) for (u, v, d) in G.edges(data=True) if 0.2 < d['weight'] <= 0.4]
e3 = [(u, v) for (u, v, d) in G.edges(data=True) if 0.4 < d['weight'] <= 0.6]
e4 = [(u, v) for (u, v, d) in G.edges(data=True) if 0.6 < d['weight'] <= 0.8]
e5 = [(u, v) for (u, v, d) in G.edges(data=True) if 0.8 < d['weight'] <= 1.0]

pos = nx.spring_layout(G, scale=10.0)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos)

# edges
nx.draw_networkx_edges(G, pos, edgelist=e1, width=1)
nx.draw_networkx_edges(G, pos, edgelist=e2, width=2)
nx.draw_networkx_edges(G, pos, edgelist=e3, width=4)
nx.draw_networkx_edges(G, pos, edgelist=e4, width=6)
nx.draw_networkx_edges(G, pos, edgelist=e5, width=10)


nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

plt.axis('off')
# plt.savefig("weighted_graph.png") # save as png
plt.show()  # display
print "done"
#nx.draw(g)
#nx.draw_random(g)
#nx.draw_circular(g)
#nx.draw_spectral(g)