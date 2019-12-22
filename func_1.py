
# coding: utf-8

# In[9]:


import pandas as pd
import os
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt


# In[10]:


#rename columns of distance dataframe
column_D = ["Node 1", "Node 2", "Distance"]
dist_df = pd.DataFrame(columns=column)


# In[4]:


#distances dataframe
with open("DIST-USA-road-d.CAL.gr") as f:
    line =f.readline()
    while line != "":
        l= line.split(" ")
        for i in range(len(l)):
            l[i] = float(l[i].strip())
        dist_df = dist_df.append(pd.DataFrame([l], columns = column), ignore_index = True)
        line =f.readline()
    

    


# In[ ]:


#rename columns of time dataframe
column_T = ["Node 1", "Node 2", "TIME"]
TIME_df = pd.DataFrame(columns=column)


# In[ ]:


#time dataframe
with open("TIME-USA-road-d.CAL.gr") as f:
    line =f.readline()
    while line != "":
        l= line.split(" ")
        for i in range(len(l)):
            l[i] = float(l[i].strip())
        time_df = time_df.append(pd.DataFrame([l], columns = column), ignore_index = True)
        line =f.readline()


# In[ ]:


#take the thershold
d = int(input("Threshold: "))


# In[ ]:


node = hou.pwd()
geo = node.geometry()
pts = geo.points()
targetpts = node.inputs()[1].geometry().points()

if len(targetpts) >= len(pts):
    from operator import itemgetter
    # add 'uniqueNeighbour' attribute
    geo.addAttrib(hou.attribType.Point, 'uniqueNeighbour', -1)

    # setup targetpts list
    targetptslist = []
    #targetptslist = [(n, targetpt.position()) for n, targetpt in enumerate(targetpts)] # short n fast version of below loop
    for n, targetpt in enumerate(targetpts):
        targetptinfo = (n, targetpt.position())
        targetptslist.append(targetptinfo)
        if hou.updateProgressAndCheckForInterrupt(): break # respect keyboard interruption

    # get the distance to every point in target geo
    for pt in pts:
        neardistlist = []
        p1 = pt.position()
        #neardistlist = [(targetptslist[i][0], (p1 - targetptslist[i][1]).length()) for i in range(len(targetptslist))] # short n fast version of below loop
        for i in range(len(targetptslist)):
            tptinfo = targetptslist[i]
            p2 = tptinfo[1]
            distance = (p1 - p2).length()
            targetinfo = (tptinfo[0], distance)
            neardistlist.append(targetinfo)
            if hou.updateProgressAndCheckForInterrupt(): break # respect keyboard interruption

        # sort the list by min distance
        #neardistlist.sort(key = lambda ptdist: ptdist[1])
        neardistlist.sort(key = itemgetter(1)) # faster than lambda sorting

        # check the neardistlist to see if this point has already been taken then remove this from the targetptslist
        nearestpt = (neardistlist[0][0])
        for j in range(len(targetptslist)):
            ptn = targetptslist[j][0]
            if ptn == nearestpt:
                del targetptslist[j]
                break
            if hou.updateProgressAndCheckForInterrupt(): break # respect keyboard interruption

        # update 'uniqueNeighbour' attribute
        pt.setAttribValue('uniqueNeighbour', nearestpt)

        if hou.updateProgressAndCheckForInterrupt(): break # respect keyboard interruption

else:
    raise hou.NodeError('Target points must be equal or more than source points!')


# In[ ]:


#graph representation

G = nx.random_geometric_graph(200, 0.125)

#create edges

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))


#color note points

node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections: '+str(len(adjacencies[1])))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

#create network graph

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()

