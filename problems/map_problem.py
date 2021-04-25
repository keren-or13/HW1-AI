from framework.graph_search.graph_problem_interface import OperatorResult
from framework import *

from typing import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os

__all__ = ['MapState', 'MapProblem']


@dataclass(frozen=True)
class MapState(GraphProblemState):
    """
    StreetsMap state is represents the current geographic location on the map.
    This location is defined by the junction index.
    """
    junction_id: int

    def __eq__(self, other):
        assert isinstance(other, MapState)
        return other.junction_id == self.junction_id

    def __hash__(self):
        return hash(self.junction_id)

    def __str__(self):
        return str(self.junction_id).rjust(5, ' ')


class MapProblem(GraphProblem):
    """
    Represents a problem on the streets map.
    The problem is defined by a source location on the map and a destination.
    """

    name = 'StreetsMap'

    def __init__(self, streets_map: StreetsMap, source_junction_id: int, target_junction_id: int, cost_func_name:str):
        initial_state = MapState(source_junction_id)
        super(MapProblem, self).__init__(initial_state)
        self.streets_map = streets_map
        self.target_junction_id = target_junction_id
        self.name += f'(src: {source_junction_id} dst: {target_junction_id})'
        assert cost_func_name in ['current_time', 'scheduled_time', 'distance']
        self.cost_func_name = cost_func_name
        self.time_to_goal_shortest_paths_based_data = None
        self.time_to_goal_history_based_data = None
    
    def set_additional_shortest_paths_based_data(self):
        # [Ex.24]: Don't edit this function! It just gives you more info.
        #           This function loads the shortest paths data from a csv file.
        #           It reads the file, and assigns the data matching to the problem's self.target_junction_id
        #           to the class variable self.time_to_goal_shortest_paths_based_data

        # set the data file path
        shortest_paths_file_path = os.path.join(Consts.DATA_PATH, 'shortest_paths.csv')
        
        # read the csv file
        df = pd.read_csv(shortest_paths_file_path) 
        
        # self-check, if the data file includes self.target_junction_id 
        assert(str(self.target_junction_id) in df.columns)

        # extract the data matching to self.target_junction_id and convert to np.array.
        # note: you can extract the data of a single column named 'name' from a pd.DataFrame 'df' by: df['name']
        data = df[str(self.target_junction_id)].to_numpy()
        
        self.time_to_goal_shortest_paths_based_data = data # assign the data

    def set_additional_history_based_data(self):
        #TODO [Ex.26]: Load additional history-based data and assign it to a class variable:
        #               (1) Load the csv file history_4_days_target_{target_id}.csv to a pd.DataFrame (pandas dataframe).
        #                   Its parent folder is framework/db/ which is the same as the parent folder of shortest_paths.csv.
        #                   Use the code in self.set_additional_shortest_paths_based_data() for help!
        #                   Each column contains data of a single day, and each row contains data of a single source: 
        #                       the data is the path cost from each source (each row) to target_id, 
        #                       based on current_speed on that day (each col).
        #               (2) Compute the mean for each source over the 4 days.
        #                   You can use pd.DataFrame.mean(axis=1) (recommended. see pandas documentation)
        #               (3) Assign self.time_to_goal_history_based_data the result of the mean.
        #                   Note: the result should be of type np.array.
        #                           you can convert a pd.DataFrame to np.array using pd.DataFrame.to_numpy()
        days_of_the_week = ['Sun', 'Mon', 'Tue', 'Wed'] # optional variable
        raise NotImplementedError  # TODO: remove this line!


        assert(type(self.time_to_goal_history_based_data) is np.ndarray) # self-check


    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[OperatorResult]:

        """
        For a given state, iterates over its successor states.
        The successor states represents the junctions to which there
        exists a road originates from the given state.
        """

        # All of the states in this problem are instances of the class `MapState`.
        assert isinstance(state_to_expand, MapState)
        y= self.cost_func_name
        # Get the junction (in the map) that is represented by the state to expand.
        junction = self.streets_map[state_to_expand.junction_id]
        for i in junction.outgoing_links:
            MapState(i.target)
            type(i.compute_current_time())
            if y=='scheduled_time':
                yield OperatorResult(MapState(i.target), i.compute_scheduled_time())
            elif y == 'current_time':
                yield OperatorResult(MapState(i.target), i.compute_current_time())
            else:
                yield OperatorResult(MapState(i.target), i.distance)

        # TODO [Ex.9]:
        #  Read the documentation of this method in the base class `GraphProblem.expand_state_with_costs()`.
        #  Finish the implementation of this method.
        #  Iterate over the outgoing links of the current junction (find the implementation of `Junction`
        #  type to see the exact field name to access the outgoing links). For each link:
        #    (1) Create the successor state (it should be an instance of class `MapState`). This state represents the
        #        target junction of the current link;
        #    (2) Yield an object of type `OperatorResult` with the successor state and the operator cost (which depends
        #        on the variable class `self.cost_func_name`). You don't have to specify the operator name here.
        #        Use:
        #           (2.1) `link.distance` where `self.cost_func_name` equals to 'distance'.
        #           (2.2) `link.compute_scheduled_time()` where `self.cost_func_name` equals to 'scheduled_time'.
        #           (2.3) `link.compute_current_time()` where `self.cost_func_name` equals to 'current_time'.
        #  Note: Generally, in order to check whether a variable is set to None you should use the expression:
        #        `my_variable_to_check is None`, and particularly do NOT use comparison (==).

        #yield OperatorResult(successor_state=MapState(self.target_junction_id), operator_cost=7)  # TODO: remove this line!

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        :return: Whether a given map state represents the destination.
        """
        assert (isinstance(state, MapState))

      #  junction = self.streets_map[state.junction_id]
       # return junction.outgoing_links is None

        # TODO [Ex.9]: modify the returned value to indicate whether `state` is a final state.
        # You may use the problem's input parameters (stored as fields of this object by the constructor).
        
        return state.junction_id == 549  # TODO: modify this!
        