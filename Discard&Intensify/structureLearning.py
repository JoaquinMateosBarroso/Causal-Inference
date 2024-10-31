from utils.graph import DGraph
import pandas as pd

class StructureLearner:
    '''Abstract class for learning the structure of a model'''
    def __init__(self, data: pd.DataFrame, initialGraph: DGraph=None,  *args,**kwargs):
        '''
            Create a structure learner
            :param data: A pandas DataFrame containing the data to learn from.
            :param initialGraph: Optional para to have an initial graph to start the learning process.
                                 If it is left as None, the learner will start with an empty graph.
        '''
        self.data = data
        if initialGraph is None:
            self.graph = DGraph(edges=[], nodes=data.columns)
        else:
            self.graph = initialGraph
        
    def learn(self) -> DGraph:
        # Do some learning
        pass

