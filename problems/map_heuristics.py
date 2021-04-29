from framework.consts import Consts
from framework.ways.streets_map import MAX_ROADS_SPEED
from framework.graph_search import *
from .map_problem import MapProblem, MapState
import numpy as np
import os


class TimeBasedAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TimeBasedAirDist'

    def estimate(self, state: GraphProblemState) -> float:
        """
        The air distance between the geographic location represented
         by `state` and the geographic location of the problem's target.

        TODO [Ex.18]: implement this method!
        Use `self.problem` to access the problem.
        Use `self.problem.streets_map` to access the map.
        Use MAX_ROADS_SPEED and MIN_ROADS_SPEED to access the max_speed 
         upper and lower bounds, respectively, of the roads speeds.
        Given a junction index, use `streets_map[junction_id]` to find the
         junction instance (of type `Junction`).
        Use the method `calc_air_distance_from()` to calculate the air
         distance between two junctions.
        """
        sol= self.problem.streets_map[self.problem.target_junction_id].calc_air_distance_from(self.problem.streets_map[state.junction_id])
        return sol/MAX_ROADS_SPEED
        assert isinstance(self.problem, MapProblem)
        assert isinstance(state, MapState)



class ShortestPathsBasedHeuristic(HeuristicFunction):
    heuristic_name = 'ShortestPathsBased'

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, MapProblem)
        assert isinstance(state, MapState)
        
        return self.problem.time_to_goal_shortest_paths_based_data[state.junction_id]

class HistoryBasedHeuristic(HeuristicFunction):
    heuristic_name = 'HistoryBased'

    def estimate(self, state: GraphProblemState) -> float:
        assert isinstance(self.problem, MapProblem)
        assert isinstance(state, MapState)

        return self.problem.time_to_goal_history_based_data[state.junction_id]

        