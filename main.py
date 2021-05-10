from framework.graph_search.astar import AStar
from framework import *
from problems import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union, Optional
import os

# Load the streets map
streets_map = StreetsMap.load_from_csv(Consts.get_data_file_path("tlv_streets_map_cur_speeds.csv"))

# Make sure that the whole execution is deterministic.
# This is important, because we expect to get the exact same results
# in each execution.
Consts.set_seed()


def plot_distance_and_expanded_wrt_weight_figure(
        problem_name: str,
        weights: Union[np.ndarray, List[float]],
        total_cost: Union[np.ndarray, List[float]],
        total_nr_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    TODO [Ex.20]: Complete the implementation of this method.
    """
    weights, total_cost, total_nr_expanded = np.array(weights), np.array(total_cost), np.array(total_nr_expanded)
    assert len(weights) == len(total_cost) == len(total_nr_expanded)
    assert len(weights) > 0
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    assert is_sorted(weights)

    fig, ax1 = plt.subplots()

    # TODO: Plot the total distances with ax1. Use `ax1.plot(...)`.
    # TODO: Make this curve colored blue with solid line style.
    # TODO: Set its label to be 'Solution cost'.
    # See documentation here:
    # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html
    # You can also Google for additional examples.
    p1, = ax1.plot(weights, total_cost, color='b', label='Soltion cost')  # TODO: pass the relevant params instead of `...`.

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Solution cost', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    # TODO: Plot the total expanded with ax2. Use `ax2.plot(...)`.
    # TODO: Make this curve colored red with solid line style.
    # TODO: Set its label to be '#Expanded states'.
    p2, = ax2.plot(weights, total_nr_expanded ,color='r' ,label='#Expanded states')  # TODO: pass the relevant params instead of `...`.

    # ax2: Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('#Expanded states', color='r')
    ax2.tick_params('y', colors='r')

    curves = [p1, p2]
    ax1.legend(curves, [curve.get_label() for curve in curves])

    fig.tight_layout()
    plt.title(f'Quality vs. time for wA* \non problem {problem_name}')
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem, n: int = 30,
                                   max_nr_states_to_expand: Optional[int] = 40_000,
                                   low_heuristic_weight: float = 0.5, high_heuristic_weight: float = 0.95):
    arr=np.linspace(low_heuristic_weight,high_heuristic_weight,n)
    cost=[]
    expended_states=[]
    weigth=[]
    for i in arr:
        uc1 = AStar(heuristic_type,i, max_nr_states_to_expand)
        res = uc1.solve_problem(problem)
        if(res.is_solution_found):
            cost.append(res.solution_g_cost)
            expended_states.append(res.nr_expanded_states)
            weigth.append(i)
    plot_distance_and_expanded_wrt_weight_figure(problem.name,weigth,cost,expended_states)

    # TODO [Ex.20]:
    #  1. Create an array of `n` numbers equally spread in the segment
    #     [low_heuristic_weight, high_heuristic_weight]
    #     (including the edges). You can use `np.linspace()` for that.
    #  2. For each weight in that array run the wA* algorithm, with the
    #     given `heuristic_type` over the given problem. For each such run,
    #     if a solution has been found (res.is_solution_found), store the
    #     cost of the solution (res.solution_g_cost), the number of
    #     expanded states (res.nr_expanded_states), and the weight that
    #     has been used in this iteration. Store these in 3 lists (list
    #     for the costs, list for the #expanded and list for the weights).
    #     These lists should be of the same size when this operation ends.
    #     Don't forget to pass `max_nr_states_to_expand` to the AStar c'tor.
    #  3. Call the function `plot_distance_and_expanded_wrt_weight_figure()`
    #     with these 3 generated lists.


# --------------------------------------------------------------------
# ------------------------ StreetsMap Problem ------------------------
# --------------------------------------------------------------------

def within_focal_h_sum_priority_function(node: SearchNode, problem: GraphProblem, solver: AStarEpsilon):
    if not hasattr(solver, '__focal_heuristic'):
        setattr(solver, '__focal_heuristic', HistoryBasedHeuristic(problem=problem))
    focal_heuristic = getattr(solver, '__focal_heuristic')
    return focal_heuristic.estimate(node.state)


def toy_map_problem_experiment():
    print()
    print('Solve the distance-based map problem.')

    # TODO [Ex.7]: Just run it and inspect the printed result.

    target_point = 549
    start_point = 82700

    dist_map_problem = MapProblem(streets_map, start_point, target_point, 'distance')

    uc = UniformCost()
    res = uc.solve_problem(dist_map_problem)
    print(res)

    # save visualization of the path
    file_path = os.path.join(Consts.IMAGES_PATH, 'UCS_path_distance_based.png')
    streets_map.visualize(path=res, file_path=file_path)


def map_problem_experiments():
    print()
    print('Solve the map problem.')

    target_point = 549
    start_point = 82700

    # TODO [Ex.12]: 1. create an instance of `MapProblem` with a current_time-based operator cost
    #           with the start point `start_point` and the target point `target_point`
    #           and name it `map_problem`.
    #       2. create an instance of `UCS`,
    #           solve the `map_problem` with it and print the results.
    #       3. save the visualization of the path in 'images/UCS_path_time_based.png'
    # You can use the code in the function 'toy_map_problem_experiment' for help.
    map_problem = MapProblem(streets_map, start_point, target_point, 'current_time')
    uc = UniformCost()
    res = uc.solve_problem(map_problem)
    print(res)

    # save visualization of the path
    file_path = os.path.join(Consts.IMAGES_PATH, 'UCS_path_time_based.png')
    streets_map.visualize(path=res, file_path=file_path)


    # TODO [Ex.16]: create an instance of `AStar` with the `NullHeuristic` (implemented in
    #       `framework\graph_search\graph_problem_interface.py`),
    #       solve the same `map_problem` with it and print the results (as before).
    # Notice: AStar constructor receives the heuristic *type* (ex: `MyHeuristicClass`),
    #         and NOT an instance of the heuristic (eg: not `MyHeuristicClass()`).
    uc1 = AStar(NullHeuristic)
    res = uc1.solve_problem(map_problem)
    print(res)

    # TODO [Ex.18]: create an instance of `AStar` with the `TimeBasedAirDistHeuristic`,
    #       and use the default value for the heuristic_weight,
    #       solve the same `map_problem` with it and print the results (as before).
    timebase = AStar(TimeBasedAirDistHeuristic)
    res = timebase.solve_problem(map_problem)
    print(res)

    # TODO [Ex.20]:
    #  1. Complete the implementation of the function
    #     `run_astar_for_weights_in_range()` (upper in this file).
    #  2. Complete the implementation of the function
    #     `plot_distance_and_expanded_wrt_weight_figure()`
    #     (upper in this file).n
    #  3. Call here the function `run_astar_for_weights_in_range()`
    #     with `TimeBasedAirDistHeuristic` and `map_problem`.

    run_astar_for_weights_in_range(TimeBasedAirDistHeuristic, map_problem)
    # TODO [Ex.24]: 1. Call the function set_additional_shortest_paths_based_data()
    #                   to set the additional shortest-paths-based data in `map_problem`.
    #                   For more info see `problems/map_problem.py`.
    #               2. create an instance of `AStar` with the `ShortestPathsBasedHeuristic`,
    #                  solve the same `map_problem` with it and print the results (as before).
    map_problem.set_additional_shortest_paths_based_data()
    shortestpath = AStar(ShortestPathsBasedHeuristic)
    res = shortestpath.solve_problem(map_problem)
    print(res)


    # TODO [Ex.25]: 1. Call the function set_additional_history_based_data()
    #                   to set the additional history-based data in `map_problem`.
    #                   For more info see `problems/map_problem.py`.
    #               2. create an instance of `AStar` with the `HistoryBasedHeuristic`,
    #                   solve the same `map_problem` with it and print the results (as before).
    map_problem.set_additional_history_based_data()
    historyheuristic = AStar(HistoryBasedHeuristic)
    res = historyheuristic.solve_problem(map_problem)
    print(res)

    # Try using A*eps to improve the speed (#dev) with a non-acceptable heuristic.
    # TODO [Ex.29]: Create an instance of `AStarEpsilon` with the `ShortestPathsBasedHeuristic`.
    #       Solve the `map_problem` with it and print the results.
    #       Use focal_epsilon=0.23, and max_focal_size=40.
    #       Use within_focal_priority_function=within_focal_h_sum_priority_function. This function
    #        (defined just above) is internally using the `HistoryBasedHeuristic`.
    shortesrastareps=AStarEpsilon(ShortestPathsBasedHeuristic,within_focal_priority_function=within_focal_h_sum_priority_function
                                  ,focal_epsilon=0.23,max_focal_size=40)
    res = shortesrastareps.solve_problem(map_problem)
    print(res)


def run_all_experiments():
    print('Running all experiments')
    toy_map_problem_experiment()
    map_problem_experiments()


if __name__ == '__main__':
    run_all_experiments()
