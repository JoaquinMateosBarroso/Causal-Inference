import networkx as nx
import matplotlib.pyplot as plt

class DGraph:
    def __init__(self, edges: list=[], adjacencyDict: dict={}, *args, **kwargs):
        '''
            Create a directed graph from either a list of edges or an adjacency list
            :param edges: A list of tuples representing the edges of the graph.
            :param adjacencyDict: A dictionary of lists representing the adjacencies 
                    of the graph, meaning that the key is the origin and each element of its' 
                    value (a set) is a destiny of an edge.
        '''
        if not edges and not adjacencyDict:
            raise ValueError("You must provide either edges or adjacencyList")
        
        if edges:
            self._edges = edges
            self._adjacencyDict = self.__createAdjacencyDict(edges)
        else:
            self._adjacencyDict = adjacencyDict
            self._edges = self.__createEdges(adjacencyDict)
    
    def getEdges(self) -> list:
        '''Return a copy of the edges of the graph'''
        return self._edges.copy()
    
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
    
    def addEdge(self, origin, destiny):
        '''Add an edge to the graph'''
        if origin not in self._adjacencyDict:
            self._adjacencyDict[origin] = set()
        self._adjacencyDict[origin].add(destiny)
        self._edges.append((origin, destiny))
    
    def removeEdge(self, origin, destiny):
        '''Remove an edge from the graph'''
        if (origin, destiny) in self._edges:
            self._edges.remove((origin, destiny))
            self._adjacencyDict[origin].remove(destiny)
    
    def reverseEdge(self, origin, destiny):
        '''Reverse an edge from the graph'''
        self.removeEdge(origin, destiny)
        self.addEdge(destiny, origin)
    
    def plot(self):
        G = nx.DiGraph(self.getEdges())

        # For a beautiful graph
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        node_color = ['green' if node == 'Survived' else 'skyblue' for node in G.nodes()]
        edge_color = ['green' if edge[1] == 'Survived' else 'black' for edge in G.edges()]

        nx.draw(G, pos, node_color=node_color, edge_color=edge_color, 
                with_labels=True, font_weight='bold')
        plt.show()

    
    def __createAdjacencyDict(self, edges: list) -> dict:
        '''Create an adjacency list from a list of edges'''
        adjacencyDict = {}
        for origin, destiny in edges:
            if origin not in adjacencyDict:
                adjacencyDict[origin] = set()
            adjacencyDict[origin].add(destiny)
        return adjacencyDict
    
    def __createEdges(self, adjacencyDict: dict) -> list:
        '''Create a list of edges from an adjacency list'''
        edges = []
        for origin in adjacencyDict:
            for destiny in adjacencyDict[origin]:
                edges.append((origin, destiny))
        return edges


    def __str__(self):
        return str(self._edges)
    
if __name__ == '__main__':
    '''Some test cases'''
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    graph = DGraph(edges)
    print(graph.isAcyclic()) # Cycle 1->2->3->4->1
    
    edges = [(1, 2), (2, 3), (3, 4)]
    graph = DGraph(edges)
    print(graph.isAcyclic()) # No cycle
    graph.plot()
    