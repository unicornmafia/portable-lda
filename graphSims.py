__author__ = 'thomas'

import os
import networkx as nx
import matplotlib.pyplot as plt
try:
   import cPickle as pickle
except:
   import pickle

threshold = 0.98

pickleFile = os.path.join(os.getcwd(), "cache/sims/sims.cos")
simsList = pickle.load(open(pickleFile, "rb"))

G = nx.Graph()

weights = []
for fileid1, fileid2, sim in simsList:
    if sim > threshold:
        G.add_edge(fileid1, fileid2, weight=sim)
    else:
        break

# e1 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.2]
# e2 = [(u, v) for (u, v, d) in G.edges(data=True) if 0.2 < d['weight'] <= 0.4]
# e3 = [(u, v) for (u, v, d) in G.edges(data=True) if 0.4 < d['weight'] <= 0.6]
# e4 = [(u, v) for (u, v, d) in G.edges(data=True) if 0.6 < d['weight'] <= 0.8]
# e5 = [(u, v) for (u, v, d) in G.edges(data=True) if 0.8 < d['weight'] <= 1.0]

# pos = nx.circular_layout(G, scale=10.0)  # positions for all nodes
#
# # nodes
# nx.draw_networkx_nodes(G, pos)
#
# # edges
# nx.draw_networkx_edges(G, pos, edgelist=e1, width=1)
# nx.draw_networkx_edges(G, pos, edgelist=e2, width=1)
# nx.draw_networkx_edges(G, pos, edgelist=e3, width=1)
# nx.draw_networkx_edges(G, pos, edgelist=e4, width=1)
# nx.draw_networkx_edges(G, pos, edgelist=e5, width=1)

nx.draw_networkx(G)


#nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

plt.axis('off')
# plt.savefig("weighted_graph.png") # save as png
plt.show()  # display
print "done"
#nx.draw(g)
#nx.draw_random(g)
#nx.draw_circular(g)
#nx.draw_spectral(g)