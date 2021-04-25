"""
 A set of utilities for using israel.csv 
 The map is extracted from the OpenStreetMap project
"""

import math
from typing import List, Tuple, Dict, Iterator
from collections import defaultdict
import itertools
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.image as img

# since the road distance is in meters, the speed is in meters per minute
kmph_to_mpm = lambda kmh: kmh * 16.667 # 1[kilometer]=1000[meter], 1[hour]=60[minute], 1[kmh]=100/60[mpm]=1.6667[mpm]
ROAD_SPEEDS = [kmph_to_mpm(kmh) for kmh in [60, 70, 80, 90, 100, 120]]
ROAD_SPEEDS_PROBS = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
MIN_ROADS_SPEED = min(ROAD_SPEEDS)
MAX_ROADS_SPEED = max(ROAD_SPEEDS)
MIN_ROADS_CURRENT_SPEED = kmph_to_mpm(20)


@dataclass(frozen=True)
class Coordinates:
    lat: float
    lon: float

def compute_air_distance_between_coordinates(point1: Coordinates, point2: Coordinates) -> float:
    """
    Computes distance in meters
    This code was borrowed from
    http://www.johndcook.com/python_longitude_latitude.html
    """

    if math.isclose(point1.lat, point2.lat) and math.isclose(point1.lon, point2.lon):
        return 0.0
    if max(abs(point1.lat - point2.lat), abs(point1.lon - point2.lon)) < 0.00001:
        return 0.001

    phi1 = math.radians(90 - point1.lat)
    phi2 = math.radians(90 - point2.lat)

    meter_units_factor = 40000 / (2 * math.pi)
    arc = math.acos(np.sin(phi1) * np.sin(phi2) * np.cos(math.radians(point1.lon) - math.radians(point2.lon))
                    + np.cos(phi1) * np.cos(phi2))
    return max(0.0, arc * meter_units_factor * 1000)


@dataclass
class Link:
    source: int
    target: int
    distance: float
    highway_type: int
    max_speed: float = 0.0
    is_toll_road: bool = False
    current_speed: float = 0.0

    @staticmethod
    def deserialize(source_idx: int, link_string: str) -> 'Link':
        link_params = [part.strip() for part in link_string.split("@") if part.strip()]
        assert len(link_params) >= 3
        target_idx = int(link_params[0])
        distance = float(link_params[1])
        highway_type = int(link_params[2])
        max_speed = float(link_params[3]) if len(link_params) > 3 else None
        is_toll_road = bool(link_params[4]) if len(link_params) > 4 else None
        current_speed = float(link_params[5]) if len(link_params) > 5 else None

        return Link(source=source_idx, target=target_idx, distance=distance, highway_type=highway_type,
                    max_speed=max_speed, is_toll_road=is_toll_road, current_speed=current_speed)

    def serialize(self) -> str:
        return f'{self.target}@{self.distance}@{self.highway_type}@{self.max_speed}@{self.is_toll_road}@{self.current_speed}'

    def get_symmetric_hash(self):
        return hash(frozenset((self.source, self.target)))
    
    # TODO [Ex.8]:
    # Compute and return the current driving time based on the given current_speed
    def compute_scheduled_time(self):
       return self.distance/self.max_speed

    # TODO [Ex.8]:
    # Compute and return the current driving time based on the given current_speed
    def compute_current_time(self):
        return self.distance / self.current_speed


@dataclass
class Junction:
    index: int
    lat: float
    lon: float
    outgoing_links: Tuple[Link, ...]
    incoming_links: Tuple[Link, ...]

    @property
    def coordinates(self) -> Coordinates:
        return Coordinates(lat=self.lat, lon=self.lon)

    # note: We explicitly define equals because we want to avoid comparing floats
    def __eq__(self, other):
        if not isinstance(other, Junction):
            return False
        return self.index == other.index

    # note: We explicitly define the hash method because we want to
    #       avoid non-deterministic hashing of a float field.
    def __hash__(self):
        return hash(self.index)

    # def __repr__(self):
    #     return f'{self.index}'

    def calc_air_distance_from(self, other_junction: 'Junction') -> float:
        assert(isinstance(other_junction, Junction))
        return compute_air_distance_between_coordinates(self.coordinates, other_junction.coordinates)
    
    @property
    def all_connected_links(self) -> Iterator[Link]:
        return itertools.chain(self.outgoing_links, self.incoming_links)

    @staticmethod
    def deserialize(serialized_junction_str: str) -> 'Junction':
        junction_idx_str, lat_str, lon_str, *serialized_links_str = (
            part.strip() for part in serialized_junction_str.split(',') if part.strip())
        junction_idx, lat, lon = int(junction_idx_str), float(lat_str), float(lon_str)
        links = tuple(Link.deserialize(junction_idx, serialized_link_str)
                      for serialized_link_str in serialized_links_str)
        return Junction(junction_idx, lat, lon, tuple(links), ())

    def serialize(self) -> str:
        serialized_links = ','.join(link.serialize() for link in self.outgoing_links)
        return f'{self.index},{self.lat},{self.lon},' + serialized_links


class StreetsMap(Dict[int, Junction]):
    """
    The StreetsMap is basically a dictionary from junction index to the Junction object.
    """

    def __init__(self, junctions_mapping: Dict[int, Junction]):
        super(StreetsMap, self).__init__(junctions_mapping)
        self.junctions_num = len(junctions_mapping)

    def get_junctions_num(self):
        return self.junctions_num

    def junctions(self) -> Iterator[Junction]:
        return iter(self.values())

    def iterlinks(self) -> Iterator[Link]:
        """iterate over all the links in the map.
        usage example:
        >>> for link in streets_map.iterlinks(): ... """
        return (link for junction in self.values() for link in junction.outgoing_links)

    def update_link_distances_to_air_distance(self):
        for link in self.iterlinks():
            link.distance = self[link.target].calc_air_distance_from(self[link.source])

    def set_incoming_links(self):
        junction_id_to_incoming_links: Dict[int, List[Link]] = defaultdict(list)
        for link in self.iterlinks():
            junction_id_to_incoming_links[link.target].append(link)
        for junction in self.junctions():
            junction.incoming_links = tuple(junction_id_to_incoming_links[junction.index])

    def remove_dangling_links(self):
        for junction in self.junctions():
            junction.outgoing_links = tuple(link for link in junction.outgoing_links if link.target in self)

    def remove_zero_distance_links(self):
        for junction in self.junctions():
            junction.outgoing_links = tuple(link for link in junction.outgoing_links if not math.isclose(link.distance, 0))

    def set_links_max_speed_and_is_toll(self, q=55):
        long_road_distance = np.percentile(a=np.array(list(link.distance for link in self.iterlinks())), q=q)
        for link in self.iterlinks():
            rnd = np.random.RandomState(link.get_symmetric_hash() % (2 ** 32))
            link.is_toll_road = rnd.choice([True, False]) if link.distance >= long_road_distance else False
            link.max_speed = MAX_ROADS_SPEED if link.is_toll_road else rnd.choice(ROAD_SPEEDS, p=ROAD_SPEEDS_PROBS)

    def set_links_current_speed(self, prob_update_current_speed=0.3):
        # set the current speed of each link, based on its max_speed, the probability for change and MIN_ROAD_CURRENT_SPEED 
        for link in self.iterlinks():
            if np.random.rand() >= prob_update_current_speed:
                link.current_speed = link.max_speed
            else:
                link.current_speed = max(np.random.rand() * link.max_speed, MIN_ROADS_CURRENT_SPEED)
                
                    
    # @staticmethod
    # def get_max_speed_bounds():
    #     return MIN_ROADS_SPEED, MAX_ROADS_SPEED

    @staticmethod
    def load_from_csv(map_filename: str) -> 'StreetsMap':
        with open(map_filename, 'rt') as map_file:
            junctions_iterator = (Junction.deserialize(row) for row in map_file)
            junction_id_to_junction_mapping = {junction.index: junction for junction in junctions_iterator}
        return StreetsMap(junction_id_to_junction_mapping)

    def write_to_csv(self, map_filename: str):
        with open(map_filename, 'w') as map_file:
            for junction in self.junctions():
                map_file.write(junction.serialize())
                map_file.write('\n')

    def visualize(self, file_path, path=None, resolution=10000):
        # plot the city map and a path (if given)

        coord2pixel = lambda cur_coord, min_coord: int((cur_coord-min_coord) * resolution)

        max_lon, max_lat = 0, 0
        min_lon, min_lat = np.inf, np.inf
        
        
        for junction in self.junctions():
            max_lon = max(junction.lon, max_lon)
            min_lon = min(junction.lon, min_lon)
            max_lat = max(junction.lat, max_lat)
            min_lat = min(junction.lat, min_lat)
        
        dy = int(np.ceil((max_lon - min_lon) * resolution))
        dx = int(np.ceil((max_lat - min_lat) * resolution))
        map_junctions = np.zeros((dx, dy))
        pixel_list = []
        for junction in self.junctions():
            x = coord2pixel(junction.lat, min_lat)
            y = coord2pixel(junction.lon, min_lon)
            map_junctions[x, y] = 0.5
            pixel_list.append((x,y))
        
        if path is None:
            img.imsave(file_path, map_junctions, cmap='magma')

        else:
            if path.solution_path is None:
                return

            junctions_coords = []
            for state in path.solution_path:
                junction_id = state.state.junction_id
                junction_coords = (self[junction_id].lat, self[junction_id].lon)
                junctions_coords.append(junction_coords)

            max_y, max_x = 0, 0
            min_y, min_x = np.inf, np.inf

            pixel_list = []
            for junction in junctions_coords:
                x = coord2pixel(junction[0], min_lat)
                y = coord2pixel(junction[1], min_lon)
                map_junctions[x, y] = 1
                pixel_list.append((x,y))
                max_x = max(max_x, x)
                min_x = min(min_x, x)
                max_y = max(max_y, y)
                min_y = min(min_y, y)

            img.imsave(file_path, map_junctions[min_x:max_x, min_y:max_y], cmap='magma')
