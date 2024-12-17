import random
import itertools
import numpy as np
import pandas as pd
import networkx as nx

def generateRandomDag(num_nodes: int, edge_prob: float) -> nx.DiGraph:
    """
    Generate a random Directed Acyclic Graph (DAG).

    Args:
        num_nodes (int): Number of nodes in the graph.
        edge_prob (float): Probability of creating an edge between two nodes (0 to 1).

    Returns:
        nx.DiGraph: A NetworkX directed acyclic graph.
    """
    G = nx.DiGraph()

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # Add edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Ensure no cycles
            if random.random() < edge_prob:
                G.add_edge(i, j)

    return G


def assignRandomProbabilities(G, num_categories: nx.DiGraph) -> None:
    """
    Assign random conditional probabilities to each node in the graph.

    Args:
        G (nx.DiGraph): A NetworkX directed acyclic graph.
        num_categories (int): Number of categories for each node.
    """
    categories = list(range(num_categories))

    for node in G.nodes():
        predecessors = list(G.predecessors(node))
        if len(predecessors) == 0:
            # If the node is a root, assign the parameters of the multinomial distribution randomly
            categoriesWeight = np.random.rand(num_categories)
            # Convert random values to probabilities
            G.nodes[node]['probabilities'] = categoriesWeight / np.sum(categoriesWeight)
        else:
            # If the node has parents, assign the conditional probabilities randomly
            G.nodes[node]['probabilities'] = dict() # Dictionary to store conditional probabilities
            for parentsCombination in itertools.product(*[categories for _ in predecessors]):
                categoriesWeight = np.random.rand(num_categories)
                G.nodes[node]['probabilities'][parentsCombination] = categoriesWeight / np.sum(categoriesWeight)

    return


def generateRandomDagAndData(numNodes, edgeProb, numCategories, numSamples):
    """
    Generates a random Directed Acyclic Graph (DAG) and uses it to create a synthetic discrete dataset.

    Parameters:
    - numNodes (int): Number of nodes in the DAG.
    - edgeProb (float): Probability of an edge existing between any two nodes.
    - numCategories (int): Number of categories for each node.
    - numSamples (int): Number of samples to generate in the dataset.

    Returns:
    - data (pd.DataFrame): Synthetic dataset generated from the DAG.
    - dag (networkx.DiGraph): Generated random DAG.
    """
    G = generateRandomDag(numNodes, edgeProb)
    
    categories = list(range(numCategories))
    
    # Assign random probabilities to the nodes
    assignRandomProbabilities(G, numCategories)
    
    # Generate data
    df = pd.DataFrame(columns=G.nodes, index=range(numSamples))

    
    def sampleNode(node) -> pd.Series:
        '''
        Obtains the values of a node in the graph by sampling from the conditional distribution given its parents.
        
        Parameters:
        - node (int): The node to sample.
        
        
        Returns:
        - pd.Series: The values of the node.
        '''
        parents = list(G.predecessors(node))
        if df[node].isnull().all(): # If the node's values have already been generated
            if len(parents) == 0: # Base case; random values for nodes with no parents
                df[node] = np.random.choice(categories,
                                            p=G.nodes[node]['probabilities'],
                                            size=numSamples)
            else: # If the node has parents, recursively generate values for the parents first
                # Obtain a list of pandas Series where each Series contains the values of the parents for a sample
                parentsValues = tuple([sampleNode(parent) for parent in parents])
                # Obtain a list of tuples where each tuple contains the values of the parents for a sample
                parentsValues = pd.Series(list(zip(*parentsValues)))
                
                # Use the parent values to generate the node's value
                nodeObtainer = lambda val: np.random.choice(categories,
                                                            p=G.nodes[node]['probabilities'][val])
                df[node] = parentsValues.apply(nodeObtainer)
                
        return df[node]
    
        
    for node in G.nodes():
        sampleNode(node)
    
    return df, G


if __name__ == '__main__':
    random.seed(0)
    # Example usage
    numNodes = 10
    edgeProb = 0.3
    numSamples = 10000
    numCategories = 2

    data, dag = generateRandomDagAndData(numNodes, edgeProb, numCategories, numSamples)

    # Display the first few rows of the generated data
    data.to_csv('data/synthetic1.csv', index=False)

    print(dag.edges)