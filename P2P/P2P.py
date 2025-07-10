import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

class Network:
    def __init__(self):
        self.num_peers = np.random.randint(50,101)
        self.adjacency_list = {i: set() for i in range(self.num_peers)}
        self.visited = set()

    def create_adjacency_list(self):
        while not all(len(neighbouring_nodes) >= 3 for neighbouring_nodes in self.adjacency_list.values()):
            i = random.randint(0,self.num_peers-1)
            j = random.randint(0,self.num_peers-1)

            if i!=j and len(self.adjacency_list[i])<6 and len(self.adjacency_list[j])<6:
                if j not in self.adjacency_list[i]:
                    self.adjacency_list[i].add(j)
                    self.adjacency_list[j].add(i)

    # To check connectivity
    def BFS(self,source):
        q = deque()
        q.append(source)
        self.visited.add(source)

        while q:
            current = q.popleft()
            for neighbour in self.adjacency_list[current]:
                if neighbour not in self.visited:
                    self.visited.add(neighbour)
                    q.append(neighbour)

    # Repeating until we get a connected graph
    def isConnected(self,source):
        connected = False
        while not connected:
            self.create_adjacency_list()
            self.visited.clear()
            count = 0
            for node in range(self.num_peers):
                if node not in self.visited:
                    count += 1
                    if count>1:
                        break
                    self.BFS(node)
            if count == 1:
                connected = True
            if not connected:
                # Resets the adjacency list
                self.adjacency_list = {i:set() for i in range(self.num_peers)}
        return True    

    def visualize(self):
        G = nx.Graph()
        for node, neighbours in self.adjacency_list.items():
            for neighbour in neighbours:
                G.add_edge(node,neighbour)
        return G
    
    # degree Constraint
    def degree_check(self):
        for node,neighbours in self.adjacency_list.items():
            if(len(neighbours)<3 or len(neighbours)>6):
                return False    
        return True
    
network = Network()

connected = network.isConnected(0)
print("The graph is Connected :",connected)
degree_checking = network.degree_check()
print("The degree constraint is followed: ",degree_checking)
G = network.visualize()

# Plotting the graph
plt.figure(figsize=(10,8))
pos = nx.spring_layout(G,seed=40)
nx.draw(G,pos,with_labels=True,node_color='skyblue',edge_color='black',node_size=500,font_size=10)
plt.title("Peer to Peer Network")
plt.savefig('P2P_Network')
plt.show()


            

        

