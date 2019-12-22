#import pacakges
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import matplotlib.patches as mpatches

#Global variables to hold the graph data
tree = [[]]
tree_cost=[[]]
neighbour_node=[]
edges_dist_cost=[]
edges_time_cost=[]
graph_edges=[]
dist_graph_file,time_graph_file,coor_file,dist_threshold,input_node=(None,)*5#objects holding the input files,threshod value
cost =0

#second functionality global
parent_node=[]
weight_matrix=[]
connected_nodes=None
graph2 = [] # default dictionary to store graph
V2=None
return_path=[]
shortest_path_list=[]
weights={}
edges=defaultdict(list)
graph_2_data=[]

#third functionality global
graph3 = defaultdict(list)
input_node_3=[]
weight_3=defaultdict(list)

#fourth functionality global
graph4 = defaultdict(list)
input_node_4=[]
weight_4=defaultdict(list)
######################find the neighbours#########################

# Recursive function to print the nodes from the input node until threshold  distance using network distance
#i.e considering all the edges have equal weight
def find_neighbour(dist, node, parent):
    global neighbour_node,graph_edges
    # Base condition threshold distance should be greater than 0
    if (dist < 0):
        return
    edge1=[]
    if parent !=-1:
        edge1.append(str(node))
        edge1.append(str(parent))
        graph_edges.append(edge1)
    #print (edge1)
    neighbour_node.append(str(node))
    # Traverse the connected nodes/adjacency list
    for i in tree[node]:
        if (i != parent):
            # node i becomes the parent of its child node
            find_neighbour(dist - 1, i, node)

# Recursive function to print the nodes from the input node until threshold  distance using distance/time function
#i.e considering weight/cost of edges based on distance /time function
def find_neighbour_with_weight(dist, node, parent,parent_array):
    global neighbour_node,graph_edges,cost
    # Base condition threshold distance should be greater than 0
    if (dist < 0):
        return
    edge1=[]
    if parent !=-1:
        print (node)
        edge1.append(str(node))
        edge1.append(str(parent))
        graph_edges.append(edge1)
    #print (edge1)
    neighbour_node.append(str(node))
    # Traverse the connected nodes/adjacency list
    #print (node)
    #print (tree[node])
    tree_updated=list(set(tree[node]))
    #print (tree_updated)
    #print (parent_array)
    for index_1,i in enumerate(tree_updated):
        for item in parent_array:
            if i==item:
                cost=0
        if (i != parent):
            # node i becomes the parent of its child node
            #calculating the cost/weight of nodes
            cost=cost+tree_cost[node][index_1]
            #print ("cost",cost)
            if  cost<dist:
                if parent!=-1:
                    if parent_array != tree_updated:
                        find_neighbour_with_weight(dist, i, node,parent_array)
                else:
                    find_neighbour_with_weight(dist, i, node, parent_array)
    #cost=0

#function to create a array/list to hold the nodes connected to the input nodes or in simple terms preparing a adjacent matrix with weight
#and also to hold the corresponding cost/weight values
def node_under_dis_with_weight(graph, node, dist, v, e,cost_array):
    global tree_cost
    for i in range(e):
        cost= cost_array[i]
        fro = graph[i][0]
        to = graph[i][1]
        # if edges_dist_cost[i]>1000:
        #print (fro)
        tree[fro].append(to)
        tree[to].append(fro)
        #if edges_dist_cost[i]>1000:
        tree_cost[fro].append(cost)
        tree_cost[to].append(cost)
        tree[fro]=list(set(tree[fro]))
        tree[to]=list(set(tree[to]))
        tree_cost[fro] = list(set(tree_cost[fro]))
        tree_cost[to] = list(set(tree_cost[to]))
    #print (tree)
    #print (tree_cost)
    find_neighbour_with_weight(dist, node, -1,tree[node])

#to prepare the adjacent matrix
#function to create a array/list to hold the nodes connected to the input nodes
def node_under_dis(graph, node, dist, v, e):
    global tree
    for i in range(e):
        fro = graph[i][0]
        to = graph[i][1]
        #if edges_dist_cost[i]>1000:
        tree[fro].append(to)
        tree[to].append(fro)
    #print (tree)
    find_neighbour(dist, node, -1)

#to form the graph for functionality 1
def create_graph():
    global tree,neighbour_node,edges_dist_cost,edges_time_cost,input_node,tree_cost
    #getting the user inputs
    print("Enter the input node")
    input_node = int(input())

    print("Enter the distance threshold")
    dist_threshold = int(input())
    function_type_str1 = "Enter the type of function t(x,y), d(x,y) or network distance \n 1 for t(x,y) \n 2 for d(x,y) \n 3 for network distance"
    print(function_type_str1)
    filter_fnc1 = input()
    stat = "finding the neighbour nodes at a distance " + str(dist_threshold) + " from the node " + str(input_node)
    print(stat)
    node_id_total=[]
    edges=[]

    #preparing the array/matrix containing nodes/edges and also corresponding cost from the graph
    with open (dist_graph_file,"r") as f:
        dist_data=f.readlines()
    for lines in dist_data:
        current_edges=[]
        if "a" in lines.split(' ')[0]:
            node_id_total.append(int(lines.split(' ')[1]))
            node_id_total.append(int(lines.split(' ')[2]))
            current_edges.append(int(lines.split(' ')[1]))
            current_edges.append(int(lines.split(' ')[2]))
            edges.append(current_edges)
            edges_dist_cost.append(int(lines.split(' ')[3]))
    with open (time_graph_file,"r") as f:
        dist_data=f.readlines()
    for lines in dist_data:
        if "a" in lines.split(' ')[0]:
            edges_time_cost.append(int(lines.split(' ')[3]))
    node_id=list(set(list(node_id_total)))
    #print (len(node_id))
    #print(edges[0])
    #print (len(edges))
    #print (max(node_id))
    if filter_fnc1=='3':
        tree=[[] for i in range (max(node_id)+1)]
        node_under_dis(edges, input_node, dist_threshold, len(node_id), len(edges))
        neighbour_node=(list(set(list(neighbour_node))))
        print("Neighbour nodes are:",neighbour_node)
        draw_graph(filter_fnc1)
    elif filter_fnc1=='2':
        tree = [[] for i in range(max(node_id) + 1)]
        tree_cost = [[] for i in range(max(node_id) + 1)]
        node_under_dis_with_weight(edges, input_node, dist_threshold, len(node_id), len(edges),edges_dist_cost)
        neighbour_node = (list(set(list(neighbour_node))))
        print("Neighbour nodes are:",neighbour_node)
        draw_graph(filter_fnc1)
    else:
        tree = [[] for i in range(max(node_id) + 1)]
        tree_cost = [[] for i in range(max(node_id) + 1)]
        node_under_dis_with_weight(edges, input_node, dist_threshold, len(node_id), len(edges), edges_time_cost)
        neighbour_node = (list(set(list(neighbour_node))))
        print("Neighbour nodes are:",neighbour_node)
        draw_graph(filter_fnc1)

#function to draw the graph /plot using matplotlib and nx
def draw_graph(filter_fnc):
    global neighbour_node,graph_edges,input_node
    print('Generating the graph for functionality find the neighbours....')
    graph1 = nx.Graph()
    nodes_color=[]
    nodes_size=[]
    #graph1.add_node(str(input_node))
    #adding the nodes size and colours
    #yellow for input node and red for connected nodes
    for item in neighbour_node:
        if item ==str(input_node):
            nodes_color.append('yellow')
            nodes_size.append(200)
        else:
            nodes_color.append('red')
            nodes_size.append(100)
    print (nodes_color)
    graph1.add_nodes_from(neighbour_node,color='red')
    graph1.add_edges_from(graph_edges,color='red')
    nx.draw(graph1, node_color=nodes_color,node_size=nodes_size,with_labels=True)
    #plt.savefig("path_graph1.png")
    #adding the legends to the plot
    red_patch = mpatches.Patch(color='red', label='Neighbour nodes')
    yellow_patch = mpatches.Patch(color='yellow', label='Input Node')
    plt.legend(handles=[red_patch, yellow_patch], loc='lower right')
    if filter_fnc=='1':

        plt.suptitle("Neighbours found with time function ")
    elif filter_fnc=='2':

        plt.suptitle("Neighbours found with distance function ")
    else:

        plt.suptitle("Neighbours found with network distance")


    plt.show()

######################find the smartest network functionality#########################

#prepare the adjacent matrix with weight
def add_edge(from_node, to_node, weight):
    global edges
        # Note: assumes edges are bi-directional
    edges[from_node].append(to_node)
    edges[to_node].append(from_node)
    weights[(from_node, to_node)] = weight
    weights[(to_node, from_node)] = weight

#processing the input data and calling the appropriate functions
def create_graph_2():
    global tree, neighbour_node, edges_dist_cost, edges_time_cost, input_node,V2,graph2,graph_2_data
    node_id_total = []
    edges = []
    #getting the inputs
    print("Enter the list of input nodes :")
    input_node_2 = []
    input_node_2_data = input()
    function_type_str = "Enter the type of function t(x,y), d(x,y) or network distance \n 1 for t(x,y) \n 2 for d(x,y) \n 3 for network distance"
    print(function_type_str)
    filter_fnc = input()
    print("Starting the functionlity to find the smartest network...please wait")
    with open(dist_graph_file, "r") as f:
        dist_data = f.readlines()
    for lines in dist_data:
        current_edges = []
        if "a" in lines.split(' ')[0]:
            node_id_total.append(int(lines.split(' ')[1]))
            node_id_total.append(int(lines.split(' ')[2]))
            current_edges.append(int(lines.split(' ')[1]))
            current_edges.append(int(lines.split(' ')[2]))
            edges.append(current_edges)
            edges_dist_cost.append(int(lines.split(' ')[3]))
    with open(time_graph_file, "r") as f:
        dist_data = f.readlines()
    for lines in dist_data:
        if "a" in lines.split(' ')[0]:
            edges_time_cost.append(int(lines.split(' ')[3]))
    node_id = list(set(list(node_id_total)))
    #print (edges)
    #preparing the array holding the input source nodes
    input_node_2_data=input_node_2_data.split(",")
    for item in input_node_2_data:
        if item !="":
            input_node_2.append(item)
    #input_node_2.remove(',')
    #print (input_node_2)
    #forming a list of paired input nodes to find the streets/edges conencting the two nodes
    combo_list = []
    for i in range(len(input_node_2)):
        #print(i)
        cnt = i
        while (cnt < len(input_node_2) - 1):
            pair_combo = []
            pair_combo.append(input_node_2[i])
            pair_combo.append(input_node_2[cnt + 1])
            combo_list.append(pair_combo)
            cnt = cnt + 1
    print("combo list : ",combo_list)


    V2 = len(node_id)+1  # No of nodes


    if filter_fnc=='3':
        for item in input_node_2:
            for data in edges:
                #print (data)
                if str(data[0]) == item or str(data[1]) == item:
                    append_data=[]
                    append_data.append(str(data[0]))
                    append_data.append(str(data[1]))
                    append_data.append(1)#adding weight as 1 to consider that all edges have equal cost since it is called based on network distance
                    graph_2_data.append(append_data)
        for data1 in graph_2_data:
            add_edge(*data1)
        #for item in graph_2_data:
            #print (item)
    elif filter_fnc=='2':
        for item in input_node_2:
            for cost,data in enumerate(edges):
                #print (data)
                if str(data[0]) == item or str(data[1]) == item:
                    append_data=[]
                    append_data.append(str(data[0]))
                    append_data.append(str(data[1]))
                    append_data.append(edges_dist_cost[cost])
                    graph_2_data.append(append_data)
        for data1 in graph_2_data:
            add_edge(*data1)
        #for item in graph_2_data:
           # print (item)
    else:
        for item in input_node_2:
            for cost,data in enumerate(edges):
                #print (data)
                if str(data[0]) == item or str(data[1]) == item:
                    append_data=[]
                    append_data.append(str(data[0]))
                    append_data.append(str(data[1]))
                    append_data.append(edges_time_cost[cost])
                    graph_2_data.append(append_data)
        for data1 in graph_2_data:
            add_edge(*data1)
        #for item in graph_2_data:
            #print (item)

    #checking if route is possible/exists between the pair of nodes formed in the above function
    for item in combo_list:
        return_path.append(find_smart_network(item[0], item[1]))
    #print (return_path)
    for item in return_path:
        if item != 'Route Not Possible':
            if len(item) > 2:
                cnt = 0
                for i in range(len(item) - 1):
                    split_list = []
                    split_list.append(item[cnt])
                    split_list.append(item[cnt + 1])
                    shortest_path_list.append(split_list)
                    cnt = cnt + 1
            else:
                shortest_path_list.append(item)
    shortest_path_list.sort() #to order te output streets/edges
    print(list(shortest_path_list for shortest_path_list, _ in itertools.groupby(shortest_path_list)))
    smartest_path=list(shortest_path_list for shortest_path_list, _ in itertools.groupby(shortest_path_list))
    draw_graph_2(input_node_2,smartest_path,filter_fnc)

#finding the smartest network which connects all the nodes using djikistras algorithm
def find_smart_network(initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        #print(current_node)
        destinations = edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

#plotiing or drawing the graph 2 using matplotlib and nx
def draw_graph_2(graph2_nodes,graph2_edges,filter_fnc):
    graph1 = nx.Graph() #ceating a simple graph using nx
    # graph1.add_node(str(input_node))
    graph1.add_nodes_from(graph2_nodes, color='red') #adding the nodes to the graph
    if filter_fnc=='1':
        graph1.add_edges_from(graph2_edges, color='red')
        plt.suptitle("Smartest Network with time function ")
    elif filter_fnc=='2':
        graph1.add_edges_from(graph2_edges, color='green')
        plt.suptitle("Smartest Network with distance function ")
    else:
        graph1.add_edges_from(graph2_edges, color='orange')
        plt.suptitle("Smartest Network with network distance")
    pos = nx.circular_layout(graph1)
    edges = graph1.edges()
    colors = [graph1[u][v]['color'] for u, v in edges]
    #weights = [graph1[u][v]['weight'] for u, v in edges]
    nx.draw(graph1,pos, edges=edges, edge_color=colors,with_labels=True)
    # plt.savefig("path_graph1.png")


    plt.show()


######################find the shortest ordered route functionality#########################

#to preapre the adjacnet matrix for graph
def add_edge_3(from_node, to_node,weight):
    global edges,weight_3
    graph3[from_node].append(to_node)
    graph3[to_node].append(from_node)
    graph3[from_node]=list(dict.fromkeys(graph3[from_node]))
    graph3[to_node] = list(dict.fromkeys(graph3[to_node]))
    weight_3[(from_node, to_node)] = weight
    weight_3[(to_node, from_node)] = weight

def printAllPathsUtil(u, d, visited, path,posi_data):

    # Mark the current node as visited and store in path
    visited[u] = True
    path.append(u)

    # If current vertex is same as destination, then print
    # current path[]
    if u == d:
        #print (path)
        data_to_append=path

        posi_data.append(data_to_append[:])
        #print(posi_data)
            #print(path)
    else:
        # If current vertex is not destination
        # Recur for all the vertices adjacent to this vertex
        for i in graph3[u]:
            if visited[i] == False:
                if i in input_node_3:
                    printAllPathsUtil(i, d, visited, path,posi_data)

    path.pop()

    visited[u] = False

def printAllPaths(s, d,V3,posi_data):

    # Mark all the vertices as not visited
    visited = [False] * (V3)

    # Create an array to store paths
    path = []

    # Call the recursive helper function to print all paths
    printAllPathsUtil(s, d, visited, path,posi_data)

def create_graph_3():

    global tree, neighbour_node, edges_dist_cost, edges_time_cost, input_node, V2, graph2, graph_2_data,input_node_3,weight_3
    node_id_total = []
    edges = []

    # get the input parameters
    input_node_3 = []
    print("Enter the source node..")
    source_node_3 = input()
    print("Enter the list of input nodes")
    input_node_3_data = input()
    function_type_str = "Enter the type of function t(x,y), d(x,y) or network distance \n 1 for t(x,y) \n 2 for d(x,y) \n 3 for network distance"
    print(function_type_str)
    filter_fnc = input()
    print ("Network analysis has been started....Please wait...")
    with open(dist_graph_file, "r") as f:
        dist_data = f.readlines()
    for lines in dist_data:
        current_edges = []
        if "a" in lines.split(' ')[0]:
            node_id_total.append(int(lines.split(' ')[1]))
            node_id_total.append(int(lines.split(' ')[2]))
            current_edges.append(int(lines.split(' ')[1]))
            current_edges.append(int(lines.split(' ')[2]))
            edges.append(current_edges)
            edges_dist_cost.append(int(lines.split(' ')[3]))
    with open(time_graph_file, "r") as f:
        dist_data = f.readlines()
    for lines in dist_data:
        if "a" in lines.split(' ')[0]:
            edges_time_cost.append(int(lines.split(' ')[3]))
    node_id = list(set(list(node_id_total)))
    #total_node_3=len(node_id)+1
    total_node_3=max(node_id)+1

    #to find connected components
    mylist_new = []
    for item in edges:
        item.sort()
        mylist_new.append(item)
    #print(mylist_new)
    mylist_new.sort()
    mylist_new=list(mylist_new for mylist_new, _ in itertools.groupby(mylist_new))
    #print (len(mylist_new))
    #print (len(edges))
    #graph = nx.from_edgelist(mylist_new)
    #nodes_connected_fromsrc=list(nx.node_connected_component(graph,int(source_node_3)))
    #print("connected components",nodes_connected_fromsrc)

    #if (list(set(input_node_3)-set(nodes_connected_fromsrc))) ==[]:
        #connected_st=True
    #else:
        #connected_st = False

    if filter_fnc == '3':
        for data in edges:
            add_edge_3(data[0],data[1],1)

    elif filter_fnc == '2':
        for index,data in enumerate(edges):
            add_edge_3(data[0],data[1],edges_dist_cost[index])

    else:
        for index,data in enumerate(edges):
            add_edge_3(data[0],data[1],edges_time_cost[index])


    #getting the input nodes and conevrting them to a list from string
    input_node_3_data=input_node_3_data.split(",")
    input_node_3.append(int(source_node_3))
    for item in input_node_3_data:
        if item !="":
            input_node_3.append(int(item))
        #input_node_2.remove(',')
    #print (input_node_3)
    #preparing a combo list or pairing the input nodes based on permutations
    combo_list3=[]
    for index3,item in enumerate(input_node_3):
        try:
            combo_lits_3_data=[]
            combo_lits_3_data.append(item)
            combo_lits_3_data.append(input_node_3[index3+1])
            combo_list3.append(combo_lits_3_data)
        except:
            pass

    #get all possible paths between source H and list of input nodes
    #print ("combo_list",combo_list3)
    possible_paths_3=[]
    for item in combo_list3:
        posi_data_1=[]
        posi_data=[]
        printAllPaths(item[0], item[1],total_node_3,posi_data)
        #print ("posi_list",posi_data)
        for item_r in posi_data:
            posi_data_1.append(item_r)
        possible_paths_3.append(posi_data)

    #get all possible paths eligible
    #print ("possible_path_list",possible_paths_3)
    data_to_be_removed=[]
    for index, item in enumerate(possible_paths_3):
        data_to_be_removed_dummy=[]
        for cnt in item:
            #print (cnt)
            diff_data=list ((set(cnt)- set(input_node_3)))
            #print ("diff_data",diff_data)
            if diff_data != []:
                data_to_be_removed_dummy.append(cnt)
            else:
                data_to_be_removed_dummy.append([])
        data_to_be_removed.append(data_to_be_removed_dummy)
    #print ("data to be removed ",data_to_be_removed)
    for index_de,item_in in enumerate(data_to_be_removed):
        for item_de in item_in:
            if item_de !=[]:
                possible_paths_3[index_de].remove(item_de)

    #print("filtered possible paths",possible_paths_3)

    #get the shortest path between source H covering all input nodes
    shortes_ordered_path_array=[]
    for index, item in enumerate(possible_paths_3):
        if filter_fnc==3:
            short_list_array_index=find_shortest_path(item,3,weight_3)
            if short_list_array_index!=[]:
                shortes_ordered_path_array.append(item[short_list_array_index])
        elif filter_fnc==2:
            short_list_array_index = find_shortest_path(item, 2,weight_3)
            if short_list_array_index != []:
                shortes_ordered_path_array.append(item[short_list_array_index])
        else:
            short_list_array_index = find_shortest_path(item, 1,weight_3)
            if short_list_array_index != []:
                shortes_ordered_path_array.append(item[short_list_array_index])
    #print (shortes_ordered_path_array)



    shortes_ordered_path_list=[]
    for item in shortes_ordered_path_array:
        if len(item)>2:
            cnt = 0
            for i in range(len(item) - 1):
                split_list = []
                split_list.append(item[cnt])
                split_list.append(item[cnt + 1])
                shortes_ordered_path_list.append(split_list)
                cnt = cnt + 1
        else:
            shortes_ordered_path_list.append(item)
    #print ("Final shortested order path is ",shortes_ordered_path_list)

    # prepare the node_list for graph
    #this is plot the intermediate nodes separately which will be traversed while reaching the destination
    #for example to if the input nodes are 1,2,3,4
    #to reach 3 from 1 if 4 has to be traversed then 4 will be a intermediate node which will again be traversed after visting 3
    shortes_ordered_node_list = []
    shortest_ordered_value_list = []
    shortested_ordered_array_new = []
    replace_no = 0
    for item_node in shortes_ordered_path_array:
        if len(item_node) > 2:
            replace_elements = item_node[1:len(item_node) - 1]
            for item_replace in replace_elements:
                str_replace = str(item_replace) + "_dummy_" + str(replace_no)
                replace_no = replace_no + 1
                item_node = [str_replace if x == item_replace else x for x in item_node]
            cnt = 0
            for i in range(len(item_node) - 1):
                split_list = []
                split_list.append(item_node[cnt])
                split_list.append(item_node[cnt + 1])
                shortested_ordered_array_new.append(split_list)
                shortes_ordered_node_list.append(split_list[0])
                shortes_ordered_node_list.append(split_list[1])
                shortest_ordered_value_list.append(str(split_list[0]).split("_")[0])
                shortest_ordered_value_list.append(str(split_list[1]).split("_")[0])
                cnt = cnt + 1
        else:
            shortested_ordered_array_new.append(item_node)
            shortes_ordered_node_list.append(item_node[0])
            shortes_ordered_node_list.append(item_node[1])
            shortest_ordered_value_list.append(item_node[0])
            shortest_ordered_value_list.append(item_node[1])

    #print("node_list", shortes_ordered_node_list)
    #print("value list", shortest_ordered_value_list)
    # input()
    #print("Final shortested order path is ", shortes_ordered_path_list)
    #print("Final shortested order node path is ", shortested_ordered_array_new)


    #find if graph is possible
    #i.e if the possible path /shortest walkable streets contains all the input nodes or not
    nodes_3 = []
    for item in shortes_ordered_path_list:
        for data in item:
            nodes_3.append(data)
    nodes_3 = list(set(nodes_3))
    possible_st=list(set(input_node_3)-set(nodes_3))
    if possible_st==[]:
        possible_status=True
    else:
        possible_status=False
    #print ("Possible status ",possible_status)

    if possible_status==True:
        draw_graph_3(shortested_ordered_array_new,input_node_3,source_node_3)
    else:
        draw_graph_3("", input_node_3, source_node_3)

#draw the graph
def draw_graph_3(edges_3_draw,graph3_nodes,input_node_draw3):

    if edges_3_draw=="":
        plt.suptitle("No Routes possible")
    #since it is a directed graph
    options = {

        'arrowstyle': '-|>',
        'arrowsize': 10,
    }
    G = nx.MultiDiGraph()#directed graph
    nodes_color = []
    nodes_size = []
    graph3_nodes_4 = []
    graph3_nodes_4_value = []
    graph3_nodes_4_final = []
    for item in edges_3_draw:
        for item2 in item:
            if "_dummy" in str(item2): #finding intermediate nodes
                graph3_nodes_4.append(item2)
                graph3_nodes_4_value.append(str(item2).split("_")[0])
            else:
                graph3_nodes_4.append(item2)
                graph3_nodes_4_value.append(item2)

    for index, item in enumerate(graph3_nodes_4):
        G.add_node(item, value=graph3_nodes_4_value[index]) #adding nodes
    for item in G.nodes:
        if str(item) == str(input_node_draw3):
            nodes_color.append('yellow')
            nodes_size.append(200)
        elif "_dummy" in str(item):
            nodes_color.append('red')
            nodes_size.append(50)
        else:
            nodes_color.append('green')
            nodes_size.append(100)
    #print(nodes_color)
    #print(G.nodes)

    G.add_edges_from(edges_3_draw)
    labels = {}
    for i in G:
        labels[i] = G.nodes[i]['value'] #adding labels to nodes
    #print(labels)
    pos = nx.circular_layout(G)
    nx.draw_circular(G, node_size=nodes_size, node_color=nodes_color, with_labels=False)
    nx.draw_networkx_labels(G, pos, labels, font_size=12)

    #adding legends to graph
    red_patch = mpatches.Patch(color='green', label='Destination nodes')
    yellow_patch = mpatches.Patch(color='yellow', label='Input Node')
    green_patch = mpatches.Patch(color='red', label='Bypass nodes')
    plt.legend(handles=[red_patch, yellow_patch,green_patch], loc='lower right')
    plt.show()

#to find the shortest walkable path between two nodes in the graph
#these two nodes are framed from the input nodes
#cost of the nodes also considered
#if it is based o network distance cost will be considered as 1
def find_shortest_path(path_array,flt_fnc3,weight_array):
    if flt_fnc3==3:
        len_array=[]
        for item_path in path_array:
            len_array.append(len(item_path))
        #print (len_array.index(min(len_array)))
        if len_array!=[]:
            return len_array.index(min(len_array))
        else:
            return []

    else:
        cost_array=[]
        for item_path in path_array:
            cost=0
            combo_lit_flt = []
            #print (item_path)
            if len(item_path) > 2:
                cnt = 0
                for i in range(len(item_path) - 1):
                    split_list = []
                    split_list.append(item_path[cnt])
                    split_list.append(item_path[cnt + 1])
                    spliy=split_list
                    combo_lit_flt.append(spliy[:])
                    cnt = cnt + 1
            else:
                combo_lit_flt.append(item_path[:])
            #print (combo_lit_flt)
            for item in combo_lit_flt:
                cost=weight_array[(item[0], item[1])]+cost

            cost_array.append(cost)
        #print (cost_array)
        if cost_array != []:
            return cost_array.index(min(cost_array))
        else:
            return []


######################find the shortest route functionality#########################

#t prepare tha adjacent matrix
def add_edge_4(from_node, to_node,weight):
    global edges,weight_4
    graph4[from_node].append(to_node)
    graph4[to_node].append(from_node)
    graph4[from_node]=list(dict.fromkeys(graph4[from_node]))
    graph4[to_node] = list(dict.fromkeys(graph4[to_node]))
    weight_4[(from_node, to_node)] = weight
    weight_4[(to_node, from_node)] = weight

def printAllPathsUtil_4(u, d, visited, path,posi_data):

    # Mark the current node as visited and store in path
    visited[u] = True
    path.append(u)

    # If current vertex is same as destination, then print
    # current path[]
    if u == d:
        #print (path)
        data_to_append=path

        posi_data.append(data_to_append[:])
        #print(posi_data)
            #print(path)
    else:
        # If current vertex is not destination
        # Recur for all the vertices adjacent to this vertex
        for i in graph4[u]:
            if visited[i] == False:
                if i in input_node_4:
                    printAllPathsUtil_4(i, d, visited, path,posi_data)

    path.pop()

    visited[u] = False

def printAllPaths_4(s, d,V3,posi_data):

    # Mark all the vertices as not visited
    visited = [False] * (V3)

    # Create an array to store paths
    path = []

    # Call the recursive helper function to print all paths
    printAllPathsUtil_4(s, d, visited, path,posi_data)

def create_graph_4():

    global tree, neighbour_node, edges_dist_cost, edges_time_cost, input_node, V2, graph2, graph_2_data,input_node_4,weight_4
    node_id_total = []
    edges = []

    # get the input parameters
    input_node_4 = []
    print("Enter the source node..")
    source_node_4 = input()
    print("Enter the list of input nodes")
    input_node_4_data = input()
    function_type_str = "Enter the type of function t(x,y), d(x,y) or network distance \n 1 for t(x,y) \n 2 for d(x,y) \n 3 for network distance"
    print(function_type_str)
    filter_fnc = input()
    print("Network analysis has been started....Please wait...")
    with open(dist_graph_file, "r") as f:
        dist_data = f.readlines()
    for lines in dist_data:
        current_edges = []
        if "a" in lines.split(' ')[0]:
            node_id_total.append(int(lines.split(' ')[1]))
            node_id_total.append(int(lines.split(' ')[2]))
            current_edges.append(int(lines.split(' ')[1]))
            current_edges.append(int(lines.split(' ')[2]))
            edges.append(current_edges)
            edges_dist_cost.append(int(lines.split(' ')[3]))
    with open(time_graph_file, "r") as f:
        dist_data = f.readlines()
    for lines in dist_data:
        if "a" in lines.split(' ')[0]:
            edges_time_cost.append(int(lines.split(' ')[3]))
    node_id = list(set(list(node_id_total)))
    #total_node_3=len(node_id)+1
    total_node_3=max(node_id)+1

    #to find connected components
    mylist_new = []
    for item in edges:
        item.sort()
        mylist_new.append(item)
    #print(mylist_new)
    mylist_new.sort()
    mylist_new=list(mylist_new for mylist_new, _ in itertools.groupby(mylist_new))
    #print (len(mylist_new))
    #print (len(edges))
    #graph = nx.from_edgelist(mylist_new)
    #nodes_connected_fromsrc=list(nx.node_connected_component(graph,int(source_node_4)))
    #print("connected components",nodes_connected_fromsrc)

    #if (list(set(input_node_4)-set(nodes_connected_fromsrc))) ==[]:
        #connected_st=True
    #else:
        #connected_st = False

    if filter_fnc == '3':
        for data in edges:
            add_edge_4(data[0],data[1],1)

    elif filter_fnc == '2':
        for index,data in enumerate(edges):
            add_edge_4(data[0],data[1],edges_dist_cost[index])

    else:
        for index,data in enumerate(edges):
            add_edge_4(data[0],data[1],edges_time_cost[index])


    input_node_4_data_1=input_node_4_data.split(",")
    input_node_4.append(int(source_node_4))
    for item in input_node_4_data_1:
        if item !="":
            input_node_4.append(int(item))
        #input_node_2.remove(',')
    #print (input_node_4)
    #print (input_node_4[-1])
    dst_node=input_node_4[-1]
    possible_combo=input_node_4
    possible_combo.pop(0)
    possible_combo.pop(-1)
    #combo_list3=[[input_node_4[0],input_node_4[-1]]]
    perms = list(itertools.permutations(possible_combo))#possible combinations of input nodes are formed
    combo_list_4 = []
    for item_per in perms:#possible combinations are added into a array with source and destination nodes
        dummy_per = []
        dummy_per.append(int(source_node_4))
        for ind_per in item_per:
            dummy_per.append(ind_per)
        dummy_per.append(dst_node)
        combo_list_4.append(dummy_per)
    #print(combo_list_4)

    input_node_4 = []
    input_node_4_data = input_node_4_data.split(",")
    input_node_4.append(int(source_node_4))
    for item in input_node_4_data:
        if item != "":
            input_node_4.append(int(item))
        # input_node_2.remove(','
    #print("input_nodes",input_node_4)
    possible_status=True
    shortes_ordered_path_list_list=[]
    shortes_ordered_path_list_list_order=[]
    replace_no=0
    for item_4 in combo_list_4:
        combo_list3=[]
        if possible_status==True:
            for index3,item_combo in enumerate(item_4):
                try:
                    combo_lits_3_data=[]
                    combo_lits_3_data.append(item_combo)
                    combo_lits_3_data.append(item_4[index3+1])
                    combo_list3.append(combo_lits_3_data)
                except:
                    pass

            #get all possible paths between source H and list of input nodes
            #print ("combo_list",combo_list3)
            possible_paths_3=[]
            for item in combo_list3:
                posi_data_1=[]
                posi_data=[]
                printAllPaths_4(item[0], item[1],total_node_3,posi_data)
                #print ("posi_list",posi_data)
                for item_r in posi_data:
                    posi_data_1.append(item_r)
                possible_paths_3.append(posi_data)

            #get all possible paths eligible
            #print ("possible_path_list",possible_paths_3)
            data_to_be_removed=[]
            for index, item in enumerate(possible_paths_3):
                data_to_be_removed_dummy=[]
                for cnt in item:
                    #print (cnt)
                    diff_data=list ((set(cnt)- set(input_node_4)))
                    #print ("diff_data",diff_data)
                    if diff_data != []:
                        data_to_be_removed_dummy.append(cnt)
                    else:
                        data_to_be_removed_dummy.append([])
                data_to_be_removed.append(data_to_be_removed_dummy)
            #print ("data to be removed ",data_to_be_removed)
            for index_de,item_in in enumerate(data_to_be_removed):
                for item_de in item_in:
                    if item_de !=[]:
                        possible_paths_3[index_de].remove(item_de)

            #print("filtered possible paths",possible_paths_3)

            #get the shortest path between source H covering all input nodes
            shortes_ordered_path_array=[]
            for index, item in enumerate(possible_paths_3):
                if filter_fnc=='3':
                    short_list_array_index=find_shortest_path_4(item,3,weight_4)
                    if short_list_array_index!=[]:
                        shortes_ordered_path_array.append(item[short_list_array_index])
                elif filter_fnc=='2':
                    short_list_array_index = find_shortest_path_4(item, 2,weight_4)
                    if short_list_array_index != []:
                        shortes_ordered_path_array.append(item[short_list_array_index])
                else:
                    short_list_array_index = find_shortest_path_4(item, 1,weight_4)
                    if short_list_array_index != []:
                        shortes_ordered_path_array.append(item[short_list_array_index])
            #print ("shortestordered_array",shortes_ordered_path_array)
            shortes_ordered_path_list=[]
            for item in shortes_ordered_path_array:
                if len(item)>2:
                    cnt = 0
                    for i in range(len(item) - 1):
                        split_list = []
                        split_list.append(item[cnt])
                        split_list.append(item[cnt + 1])
                        shortes_ordered_path_list.append(split_list)
                        cnt = cnt + 1
                else:
                    shortes_ordered_path_list.append(item)


            #prepare the node_list for graph
            shortes_ordered_node_list=[]
            shortest_ordered_value_list=[]
            shortested_ordered_array_new=[]
            for item_node in shortes_ordered_path_array:
                if len(item_node)>2:
                    replace_elements=item_node[1:len(item_node)-1]
                    for item_replace in replace_elements:
                        str_replace=str(item_replace)+"_dummy_"+str(replace_no)
                        replace_no=replace_no+1
                        item_node = [str_replace if x == item_replace else x for x in item_node]
                    cnt = 0
                    for i in range(len(item_node) - 1):
                        split_list = []
                        split_list.append(item_node[cnt])
                        split_list.append(item_node[cnt + 1])
                        shortested_ordered_array_new.append(split_list)
                        shortes_ordered_node_list.append(split_list[0])
                        shortes_ordered_node_list.append(split_list[1])
                        shortest_ordered_value_list.append(str(split_list[0]).split("_")[0])
                        shortest_ordered_value_list.append(str(split_list[1]).split("_")[0])
                        cnt = cnt + 1
                else:
                    shortested_ordered_array_new.append(item_node)
                    shortes_ordered_node_list.append(item_node[0])
                    shortes_ordered_node_list.append(item_node[1])
                    shortest_ordered_value_list.append(item_node[0])
                    shortest_ordered_value_list.append(item_node[1])

            #print ("node_list",shortes_ordered_node_list)
            #print ("value list",shortest_ordered_value_list)
            #input()
            #print ("Final shortested order path is ",shortes_ordered_path_list)
            print ("Final shortested order node path is ",shortested_ordered_array_new)
            shortes_ordered_path_list_list.append(shortes_ordered_path_list)
            shortes_ordered_path_list_list_order.append(shortested_ordered_array_new)
            #find if graph is possible
            nodes_3 = []
            for item in shortes_ordered_path_list:
                for data in item:
                    nodes_3.append(data)
            nodes_3 = list(set(nodes_3))
            possible_st=list(set(input_node_4)-set(nodes_3))
            if possible_st==[]:
                possible_status=True
            else:
                possible_status=False
            #print ("Possible status ",possible_status)

    if possible_status==True:
        #print (shortes_ordered_path_list_list)
        if filter_fnc == '3':
            #print ("function3")
            short_list_array_index_final = find_shortest_final_path_4(shortes_ordered_path_list_list, 3, weight_4)
            if short_list_array_index_final != []:
                shortes_ordered_path_list_final=(shortes_ordered_path_list_list[short_list_array_index_final])
                shortes_ordered_path_list_final_order=(shortes_ordered_path_list_list_order[short_list_array_index_final])
        elif filter_fnc == '2':
            short_list_array_index_final = find_shortest_final_path_4(shortes_ordered_path_list_list, 2, weight_4)
            if short_list_array_index_final != []:
                shortes_ordered_path_list_final = (shortes_ordered_path_list_list[short_list_array_index_final])
                shortes_ordered_path_list_final_order = (shortes_ordered_path_list_list_order[short_list_array_index_final])
        else:
            #print ("function else")
            short_list_array_index_final = find_shortest_final_path_4(shortes_ordered_path_list_list, 1, weight_4)
            if short_list_array_index_final != []:
                shortes_ordered_path_list_final = (shortes_ordered_path_list_list[short_list_array_index_final])
                shortes_ordered_path_list_final_order = (shortes_ordered_path_list_list_order[short_list_array_index_final])
        #print ("finalised walkable distance: ",shortes_ordered_path_list_final)
        #print ("finalised walkable ordered distance: ",shortes_ordered_path_list_final_order)

        #draw_graph_4(shortes_ordered_path_list_final,input_node_4,source_node_4)
        draw_graph_4(shortes_ordered_path_list_final_order, input_node_4, source_node_4)
    else:
        draw_graph_4("", input_node_4, source_node_4)

#to find the shortest walkable path from all possible shortest paths
#_ returned with the combo pairs
def find_shortest_final_path_4(posi_array_list,flt_fct,cost_array):
    if flt_fct==3:
        total_net_array=[]
        #print ("posi array list",posi_array_list)
        for item1 in posi_array_list:
            #print (item1)
            total_length=0
            for item2 in item1:
                #print(item2)
                total_length=total_length+len(item2)
                #print (total_length)
            total_net_array.append(total_length)
        #print (total_net_array)
        if total_net_array!=[]:
            return total_net_array.index(min(total_net_array))
        else:
            return []
    else:
        cost_array_final=[]
        for item1 in posi_array_list:
            cost_final=0
            combo_lit_flt_final = []
            #print (item_path)
            for item2 in item1:
                if len(item2) > 2:
                    cnt = 0
                    for i in range(len(item2) - 1):
                        split_list = []
                        split_list.append(item2[cnt])
                        split_list.append(item2[cnt + 1])
                        spliy=split_list
                        combo_lit_flt_final.append(spliy[:])
                        cnt = cnt + 1
                else:
                    combo_lit_flt_final.append(item2[:])
            #print (combo_lit_flt)
                for item in combo_lit_flt_final:
                    cost_final=cost_array[(item[0], item[1])]+cost_final

            cost_array_final.append(cost_final)
        #print (cost_array)
        if cost_array_final != []:
            return cost_array_final.index(min(cost_array_final))
        else:
            return []

#draw the directed graph
def draw_graph_4(edges_3_draw,graph3_nodes,input_node_draw3):

    if edges_3_draw=="":
        plt.suptitle("No routes possible")
    options = {

        'arrowstyle': '-|>',
        'arrowsize': 10,
    }
    G = nx.MultiDiGraph()
    nodes_color=[]
    nodes_size=[]
    graph3_nodes_4=[]
    graph3_nodes_4_value=[]
    graph3_nodes_4_final=[]
    for item in edges_3_draw:
        for item2 in item:
            if "_dummy" in str(item2):
                graph3_nodes_4.append(item2)
                graph3_nodes_4_value.append(str(item2).split("_")[0])
            else:
                graph3_nodes_4.append(item2)
                graph3_nodes_4_value.append(item2)

    for index,item in enumerate(graph3_nodes_4):
        G.add_node(item,value=graph3_nodes_4_value[index])
    for item in G.nodes:
        if str(item) ==str(input_node_draw3):
            nodes_color.append('yellow')
            nodes_size.append(200)
        elif "_dummy" in str(item):
            nodes_color.append('red')
            nodes_size.append(50)
        else:
            nodes_color.append('green')
            nodes_size.append(100)
    #print (nodes_color)
    #print (G.nodes)

    G.add_edges_from(edges_3_draw)
    labels = {}
    for i in G:
        labels[i] = G.nodes[i]['value']
    #print (labels)
    pos = nx.circular_layout(G)
    nx.draw_circular(G, node_size=nodes_size, node_color=nodes_color, with_labels=False)
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    #G.add_nodes_from(graph3_nodes_4_final)
    #nx.draw_networkx(G,node_color=nodes_color,node_size=nodes_size, arrows=True, **options)
    red_patch = mpatches.Patch(color='green', label='Destination nodes')
    yellow_patch = mpatches.Patch(color='yellow', label='Input Node')
    green_patch = mpatches.Patch(color='red', label='Bypass nodes')
    plt.legend(handles=[red_patch, yellow_patch, green_patch], loc='lower right')
    plt.show()


#to find the shortest walkable path between two nodes in the graph
#these two nodes are framed from the input nodes
#cost of the nodes also considered
#if it is based o network distance cost will be considered as 1
def find_shortest_path_4(path_array,flt_fnc3,weight_array):
    if flt_fnc3==3:
        len_array=[]
        for item_path in path_array:
            len_array.append(len(item_path))
        #print (len_array.index(min(len_array)))
        if len_array!=[]:
            return len_array.index(min(len_array))
        else:
            return []

    else:
        cost_array=[]
        for item_path in path_array:
            cost=0
            combo_lit_flt = []
            #print (item_path)
            if len(item_path) > 2:
                cnt = 0
                for i in range(len(item_path) - 1):
                    split_list = []
                    split_list.append(item_path[cnt])
                    split_list.append(item_path[cnt + 1])
                    spliy=split_list
                    combo_lit_flt.append(spliy[:])
                    cnt = cnt + 1
            else:
                combo_lit_flt.append(item_path[:])
            #print (combo_lit_flt)
            for item in combo_lit_flt:
                cost=weight_array[(item[0], item[1])]+cost

            cost_array.append(cost)
        #print (cost_array)
        if cost_array != []:
            return cost_array.index(min(cost_array))
        else:
            return []

########## main function call #########
if __name__ == "__main__":
    start_string="Enter any one of the number to select the corresponding functionality "+'\n'+'1 for Find Neighbours'+'\n'+'2 for Find Smartest Network'+'\n'+'3 for Find shortest ordered path'+'\n'+'4 for Find shortest path'
    print (start_string)
    functionlity=input()
    if functionlity=='1':
        print ("Starting the functionlity to find the  neighbours")
        print("Provide the distance graph file")
        dist_graph_file = input()
        print("Provide the time graph file")
        time_graph_file = input()
        create_graph()

    elif functionlity=='2':

        print("Provide the distance graph file")
        dist_graph_file = input()
        print("Provide the time graph file")
        time_graph_file = input()
        create_graph_2()
        #find_smart_network()
    elif functionlity=='3':
        print("Starting the functionlity to find the shortest ordered route from H source to Pn destination")
        print("Provide the distance graph file")
        dist_graph_file = input()
        print("Provide the time graph file")
        time_graph_file = input()
        create_graph_3()
    else:
        print("Starting the functionlity to find the shortest route from H source to Pn destination")
        print("Provide the distance graph file")
        dist_graph_file = input()
        print("Provide the time graph file")
        time_graph_file = input()
        create_graph_4()
