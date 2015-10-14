__author__ = 'thomas'
"""
An example using Graph as a weighted network.
"""
try:
    import matplotlib.pyplot as plt
except:
    raise

import networkx as nx

G=nx.Graph()

G.add_edge('a','b',weight=0.6)
G.add_edge('a','c',weight=0.2)
G.add_edge('c','d',weight=0.1)
G.add_edge('c','e',weight=0.7)
G.add_edge('c','f',weight=0.9)
G.add_edge('a','d',weight=0.3)

e1=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.2]
e2=[(u,v) for (u,v,d) in G.edges(data=True) if 0.2 < d['weight'] <= 0.4]
e3=[(u,v) for (u,v,d) in G.edges(data=True) if 0.4 < d['weight'] <= 0.6]
e4=[(u,v) for (u,v,d) in G.edges(data=True) if 0.6 < d['weight'] <= 0.8]
e5=[(u,v) for (u,v,d) in G.edges(data=True) if 0.8 < d['weight'] <= 1.0]

pos=nx.spring_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,node_size=700)

# edges
nx.draw_networkx_edges(G,pos,edgelist=e1,
                    width=1)
nx.draw_networkx_edges(G,pos,edgelist=e2,
                    width=2)
nx.draw_networkx_edges(G,pos,edgelist=e3,
                    width=4)
nx.draw_networkx_edges(G,pos,edgelist=e4,
                    width=6)
nx.draw_networkx_edges(G,pos,edgelist=e5,
                    width=8)



# labels
nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

plt.axis('off')
plt.savefig("weighted_graph.png") # save as png
plt.show() # display