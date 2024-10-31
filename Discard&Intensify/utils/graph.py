import random
import networkx as nx
from typing import TypeVar
import matplotlib.pyplot as plt

T = TypeVar('T') # Node Type
class DGraph:
    def __init__(self, edges: set[tuple[T]]=None, adjacencyDict: dict[T, list[T]]=None, 
                 nodes: set[T]=None, *args, **kwargs):
        '''
            Create a directed graph from either a list of edges or an adjacency list. If 
            no parameter is provided, the graph will be empty.
            :param edges: A set of tuples representing the edges of the graph.
            :param adjacencyDict: A dictionary of lists representing the adjacencies 
                    of the graph, meaning that the key is the origin and each element of its' 
                    value (a set) is a destiny of an edge.
        '''
        # If edges are provided, create the adjacency list
        if edges:
            self._edges = edges
            self._adjacencyDict = self.__createAdjacencyDict(edges)
        elif adjacencyDict: # If an adjacency list is provided, create the edges
            self._adjacencyDict = adjacencyDict
            self._edges = self.__createEdges(adjacencyDict)
        else: # If no edges are provided, create an empty graph
            self._edges = set()
            self._adjacencyDict = {}
        
        # Create a set of nodes from the edges and given nodes, in case there were any
        self._nodes = set()
        if nodes is not None:
            self._nodes = self._nodes.union(nodes.copy())
        self._nodes = self._nodes.union({node for edge in self._edges for node in edge})
    
    def getEdges(self) -> list:
        '''Return a copy of the edges of the graph'''
        return list(self._edges.copy())
    
    def getAdjacencyDict(self) -> dict:
        '''Return a copy of the adjacency list of the graph'''
        return self._adjacencyDict.copy()
    
    def isAcyclic(self) -> bool:
        '''Check if the graph is acyclic'''
        # Sets to keep track of visited nodes and nodes in the recursion stack
        visited = set()
        recStack = set()

        # Define a helper function for DFS
        def dfs(v):
            # Mark the current node as visited and add it to the recursion stack
            visited.add(v)
            recStack.add(v)

            # Recur for all neighbors of the current node
            for neighbor in self._adjacencyDict.get(v, []):
                # If neighbor is in the recursion stack, we have a cycle
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in recStack:
                    return True

            # Remove the node from the recursion stack
            recStack.remove(v)
            return False

        # Perform DFS for every unvisited node in the graph
        for node in self._adjacencyDict:
            if node not in visited:
                if dfs(node):
                    return False

        return True
    
    def addEdge(self, origin: T=None, destiny: T=None):
        '''Add an edge to the graph.
           If no edges are provided, there will be chosen a random edge to add.'''
        if origin is None or destiny is None:
            origin, destiny = random.sample(list(self._nodes), 2)
            
        if origin not in self._adjacencyDict:
            self._adjacencyDict[origin] = set()
        self._adjacencyDict[origin].add(destiny)
        self._edges.add((origin, destiny))
        
        self._nodes = self._nodes.union({origin, destiny})
    
    def removeEdge(self, origin=None, destiny=None):
        '''Remove an edge from the graph.
           If no edges are provided, there will be chosen a random edge to remove.'''
        if origin is None or destiny is None:
            origin, destiny = random.choice(list(self._edges))
            
        if (origin, destiny) in self._edges:
            self._edges.remove((origin, destiny))
            self._adjacencyDict[origin].remove(destiny)
        else:
            raise ValueError(f"Edge ({origin}, {destiny}) not found in the graph")
    
    def reverseEdge(self, origin=None, destiny=None):
        '''Reverse an edge from the graph.
           If no edges are provided, there will be chosen a random edge to reverse.'''
        if origin is None or destiny is None:
            origin, destiny = random.choice(list(self._edges))
        self.removeEdge(origin, destiny)
        self.addEdge(destiny, origin)
    
    def copy(self):
        '''Return a copy of the graph'''
        return DGraph(edges=self._edges.copy(), nodes=self._nodes.copy())
    
    def plot(self, objectives=[]):
        G = nx.DiGraph(self.getEdges())
        for node in self._nodes:
            if node not in G.nodes():
                G.add_node(node)

        # For a beautiful graph
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        node_color = ['green' if node in objectives else 'skyblue' for node in G.nodes()]
        edge_color = ['green' if edge[1] in objectives else 'black' for edge in G.edges()]

        nx.draw(G, pos, node_color=node_color, edge_color=edge_color,
                with_labels=True, font_weight='bold')
        plt.show()
    
    def __createAdjacencyDict(self, edges: set) -> dict:
        '''Create an adjacency list from a list of edges'''
        adjacencyDict = {}
        for origin, destiny in edges:
            if origin not in adjacencyDict:
                adjacencyDict[origin] = set()
            adjacencyDict[origin].add(destiny)
        return adjacencyDict
    
    def __createEdges(self, adjacencyDict: dict) -> list:
        '''Create a list of edges from an adjacency list'''
        edges = set()
        for origin in adjacencyDict:
            for destiny in adjacencyDict[origin]:
                edges.add((origin, destiny))
        return edges


    def __str__(self):
        return str(self._edges)

def createFullyConnectedDGraph(nodes: list) -> DGraph:
    '''Create a fully connected directed graph with given nodes'''
    edges = {(i, j) for i in nodes for j in nodes if i != j}
    return DGraph(edges)
    
if __name__ == '__main__':
    '''Some test cases'''
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    graph = DGraph(edges)
    print(graph.isAcyclic()) # Cycle 1->2->3->4->1
    
    edges = [(1, 2), (2, 3), (3, 4)]
    graph = DGraph(edges)
    print(graph.isAcyclic()) # No cycle
    graph.plot(objectives=[1, 4])
    