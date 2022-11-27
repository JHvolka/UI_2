import itertools
import math
import random
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont, ImageTk
import os
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
import time
import pandas as pd


class PopulationSizeError(Exception):
    """Error thrown when population size is not the same as expected"""

    def __init__(self, size, message="Population size not correct"):
        self.size = size
        self.message = message
        super().__init__(self.message)


@dataclass
class City:
    """Class holding a single city coordinates and metadata.

    Args:
        x: X coordinate.
        y: Y coordinate.
        original_pos: Position original path permutation.
        """
    x: float
    y: float
    original_pos: int
    identifier: int = field(default_factory=itertools.count().__next__, init=False)

    def __str__(self):
        return f"City {self.identifier}"

    def __repr__(self):
        return f"{self.identifier} ({self.x:.1f}, {self.y:.1f})"


class Map:
    newid = itertools.count().__next__

    def __init__(self, number_of_cities: int):
        """Holds a map of cities that can be optimized
        :type number_of_cities: int
        :param number_of_cities: Number of cities that will be generated.
        """
        self.id = Map.newid()
        self.cities: list[City] = []
        self.number_of_cities = number_of_cities
        self.fitness = None

    def __hash__(self):
        ret_val = 0
        for idx, city in enumerate(self.cities):
            ret_val += ret_val + (city.original_pos * idx)
        return ret_val

    def get_num_of_cities(self) -> int:
        """
        :return: number of cities on the map
        """
        return self.number_of_cities

    def generate_cities(self):
        """
        Removes existing cities from map if they exist and generates new random ones.
        """
        self.cities = []
        for i in range(0, self.number_of_cities):
            self.cities.append(City(random.uniform(-200.0, 200.0), random.uniform(-200.0, 200.0), i))

    def set_cities(self, city_list: list[City]):
        """
        Sets cities on the map in a given order
        :param city_list: list of cities to set the map to. Order is preserved

        :raises PopulationSizeError
        """
        if len(city_list) != self.number_of_cities:
            raise PopulationSizeError
        self.cities = city_list

    def get_fitness(self) -> float:
        """ Returns fitness of path.

        Fitness is cached for repeated queries

        :rtype: float
        :return: fitness of path between cities
        """
        if self.fitness is not None:
            return self.fitness

        self.fitness = 0
        for idx, val in enumerate(self.cities):
            self.fitness += math.dist((self.cities[idx].x, self.cities[idx].y),
                                      (self.cities[idx - 1].x, self.cities[idx - 1].y))
        return self.fitness

    def draw(self, path: str = "images/",
             append: str = "",
             background: (int, int, int) = (0x15, 0x15, 0x15),
             scaling: int = 3,
             offset: int = 50,
             map_radius: int = 200,
             city_radius: int = 10,
             multiplier: int = 3,
             font_size: int = 20,
             ) -> str:
        """
        Renders and image of map with paths between cities.

        Resulting image is saved as file. Path and filename are returned as string.

        :param path: path to new image file
        :param append: appendix to file name
        :param background: background colour
        :param scaling: resolution multiplier for downscaling
        :param offset: border length in pixels
        :param map_radius: half the length of map edge
        :param city_radius: radius of a city dot
        :param multiplier: multiplies map diameter to get picture size
        :param font_size: size of font
        :return: path to resulting file

        :type path: str
        :type append: int
        :type background: (int,int,int)
        :type scaling: int
        :type offset: int
        :type map_radius: int
        :type city_radius: int
        :type multiplier: int
        :type font_size: int
        :rtype str
        """
        offset = offset * scaling
        city_radius = city_radius * scaling
        multiplier = multiplier * scaling
        font_size = font_size * scaling

        image_size = map_radius * 2 * multiplier + 2 * offset
        image = Image.new('RGB', (image_size, image_size), background)

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", font_size)

        for idx, val in enumerate(self.cities):
            draw.line(((int(self.cities[idx].x) + map_radius) * multiplier + offset,
                       (int(self.cities[idx].y) + map_radius) * multiplier + offset,
                       (int(self.cities[idx - 1].x) + map_radius) * multiplier + offset,
                       (int(self.cities[idx - 1].y) + map_radius) * multiplier + offset),
                      fill=(0, 235, 180), width=scaling)
        for val in self.cities:
            draw.ellipse(
                ((int(val.x) + map_radius) * multiplier + offset - city_radius,
                 (int(val.y) + map_radius) * multiplier + offset - city_radius,
                 (int(val.x) + map_radius) * multiplier + offset + city_radius,
                 (int(val.y) + map_radius) * multiplier + offset + city_radius),
                outline=(0, 255, 200), fill=(0, 130, 80))
            draw.text(
                ((int(val.x) + map_radius) * multiplier + offset, (int(val.y) + map_radius) * multiplier + offset + 30),
                font=font,
                text=val.__repr__())

        # Create path to images if it does not exist
        exist = os.path.exists(path)
        if not exist:
            os.makedirs(path)

        image = image.resize((image_size // scaling, image_size // scaling), resample=3)

        #
        image.save(f'{path}map_{append}.png')
        # with open(f'{path}map_{generation}_{number}_{self.id}.png', 'wb') as f:
        #     w = png.Writer(width, height, greyscale=False)
        #     w.write(f, img)
        return f'{path}map_{append}.png'

    def random_neighbour(self):
        """
        Generated a ranodm neighbour.

        A random neighbour is generated by selecting two random cities and reversing the path between them
        (including them)

        :return: new map with a small random mutation
        """
        m = deepcopy(self)

        c_1 = int(random.uniform(0, self.number_of_cities))
        c_2 = int(random.uniform(0, self.number_of_cities))

        # noinspection PyUnreachableCode
        if True:  # random.uniform(0, 1) > 0.0:
            # reverse
            if c_1 > c_2:
                temp = m.cities[c_2::-1] + m.cities[self.get_num_of_cities():c_1 - 1:-1] + m.cities[c_2 + 1:c_1]
            else:
                if c_1 == 0:
                    temp = m.cities[c_2:c_1:-1] + m.cities[c_1:c_1 + 1] + m.cities[c_2 + 1:]
                else:
                    temp = m.cities[:c_1] + m.cities[c_2:c_1 - 1:-1] + m.cities[c_2 + 1:]
        else:
            # TODO other mutation
            pass
        m.cities = temp
        m.fitness = None
        return m

    def random_neighbourhood(self, count: int = 20):
        return [self.random_neighbour() for n in range(0, count)]


def probability_function(len_orig, len_new, temp):
    """ Calculates the probability new map should be chosen

    :param len_orig: length of original path
    :param len_new: length of new path
    :param temp: temperature
    :return: probability of choosing new path
    """
    if len_orig == len_new:
        return 0
    elif len_orig > len_new:
        return 1
    try:
        return math.exp((len_orig - len_new) / temp)
    except OverflowError:
        return 1


class Schedules(Enum):
    EXPONENTIAL = 1
    QUADRATIC = 2
    LINEAR = 3


def simulated_annealing(m: Map, **kwargs):
    """
    Algorithm called simulated annealing for a path between cities on a map.



    :param m: map
    :param kwargs:
    :return:
    """
    max_states_multiplier: int = kwargs.get("max_states_multiplier", 30)
    neighbour_count_multiplier: float = kwargs.get("neighbour_count_multiplier", 2)
    initial_temp_multiplier: float = kwargs.get("initial_temp_multiplier", 500)
    cooling_factor: float = kwargs.get("cooling_factor", 0.99)
    schedule: Schedules = kwargs.get("schedule", Schedules.EXPONENTIAL)

    initial_temperature: float = initial_temp_multiplier * m.get_num_of_cities()
    temperature = initial_temperature
    temperature_map = []
    fitness_map = []
    state_count = 0
    for i in range(0, max_states_multiplier * m.get_num_of_cities()):
        for j in range(0, int(m.get_num_of_cities() * neighbour_count_multiplier)):
            # Calculate temperature, according to schedule
            if schedule == Schedules.EXPONENTIAL:
                temperature = initial_temperature * math.pow(cooling_factor, state_count)
            elif schedule == Schedules.QUADRATIC:
                temperature = initial_temperature / (1 + cooling_factor * state_count * state_count)
            elif schedule == Schedules.LINEAR:
                temperature = initial_temperature / (1 + cooling_factor * state_count)

            state_count += 1
            neighbour = m.random_neighbour()
            if probability_function(m.get_fitness(), neighbour.get_fitness(), temperature) > random.uniform(0.0, 1.0):
                m = neighbour
                temperature_map.append(temperature)
                fitness_map.append(m.get_fitness())
                break
            else:
                temperature_map.append(temperature)
                fitness_map.append(m.get_fitness())
        else:
            break

    return m, temperature_map, fitness_map


def test_simulated_annealing(m: Map,
                             dataframe: pd.DataFrame,
                             name: str = "",
                             max_states_multiplier: int = 30,
                             neighbour_count_multiplier: float = 2,
                             initial_temp_multiplier: float = 500,
                             cooling_factor: float = 0.99,
                             schedule: Schedules = Schedules.EXPONENTIAL):
    start = time.time()
    simulated, temp_map, fit_map = \
        simulated_annealing(m,
                            max_states_multiplier=max_states_multiplier,
                            schedule=schedule,
                            cooling_factor=cooling_factor,
                            neighbour_count_multiplier=neighbour_count_multiplier,
                            initial_temp_multiplier=initial_temp_multiplier)
    end = time.time()

    simulated.draw(append=str(m.get_num_of_cities()) + name)
    new_df = pd.DataFrame({"name": [name],
                           "time": [end - start],
                           "start_fitness": [m.get_fitness()],
                           "end_fitness": [simulated.get_fitness()]})
    dataframe = pd.concat([dataframe, new_df], axis=0, ignore_index=True)

    return dataframe, temp_map, fit_map


def annealing_test(city_count: int = 30, print_table: bool = False, output_images: bool = True):
    map_orig = Map(city_count)
    map_orig.generate_cities()

    df = pd.DataFrame(data={"name": [],
                            "time": [],
                            "start_fitness": [],
                            "end_fitness": []})

    map_orig.draw(append="orig")

    df, temp_map_quad, fit_map_quad = test_simulated_annealing(map_orig,
                                                               df,
                                                               "Quadratic",
                                                               schedule=Schedules.QUADRATIC,
                                                               cooling_factor=0.85,
                                                               initial_temp_multiplier=1000)

    df, temp_map_lin, fit_map_lin = test_simulated_annealing(map_orig,
                                                             df,
                                                             "Linear",
                                                             schedule=Schedules.LINEAR,
                                                             initial_temp_multiplier=800)

    df, temp_map_exp, fit_map_exp = test_simulated_annealing(map_orig,
                                                             df,
                                                             "Exponential",
                                                             initial_temp_multiplier=400,
                                                             cooling_factor=0.99)

    if output_images:
        sns.set(rc={'figure.figsize': (10, 10)})
        sns.lineplot(data=temp_map_quad, errorbar=None, estimator=None, label="quadratic")
        sns.lineplot(data=temp_map_lin, errorbar=None, estimator=None, label="linear")
        ax = sns.lineplot(data=temp_map_exp, errorbar=None, estimator=None, label="exponential")
        ax.set(xlabel='Iterations', ylabel='Temperature')
        ax.set(yscale='log')
        plt.savefig("temp_map_" + str(city_count) + ".png")
        plt.clf()
        sns.lineplot(data=fit_map_quad, errorbar=None, estimator=None, label="quadratic")
        sns.lineplot(data=fit_map_lin, errorbar=None, estimator=None, label="linear")
        ax = sns.lineplot(data=fit_map_exp, errorbar=None, estimator=None, label="exponential")
        ax.set(xlabel='Iterations', ylabel='Fitness')
        plt.savefig("fit_map_" + str(city_count) + ".png")
        plt.clf()

    df.to_csv("simulated_annealing.csv")

    if print_table:
        print(df)
    return df


def tabu_search(m: Map, max_tabu_size: int = 500,
                max_iter_count_multiplier: int = 200,
                neighbour_count: int = 30,
                max_non_improvement_count_multiplier: float = 2) -> (Map, list):
    s_best: Map = m
    best_candidate: Map = m
    tabu_list: {Map} = {m: None}  # Dict and not set, to preserve order

    fit_map_best = []
    fit_map_immediate = []

    since_improvement = 0

    for i in range(0, (max_iter_count_multiplier * m.get_num_of_cities())):
        if pow(m.get_num_of_cities(), max_non_improvement_count_multiplier) <= since_improvement:
            break
        # print(since_improvement)
        since_improvement += 1
        neighbourhood: [] = best_candidate.random_neighbourhood(neighbour_count)
        best_candidate = neighbourhood[0]
        for candidate in neighbourhood:
            if ((candidate not in tabu_list) and candidate.get_fitness() <= best_candidate.get_fitness()) or \
                    ((candidate in tabu_list) and candidate.get_fitness() * 1.05 <= best_candidate.get_fitness()):
                best_candidate = candidate

        if best_candidate.get_fitness() < s_best.get_fitness():
            since_improvement = 0
            s_best = best_candidate
        tabu_list[best_candidate] = None
        if len(tabu_list) > max_tabu_size:
            tabu_list.pop(next(iter(tabu_list)))
        fit_map_immediate.append(best_candidate.get_fitness())
        fit_map_best.append(s_best.get_fitness())

    return s_best, fit_map_best, fit_map_immediate


def tabu_test(city_count: int):
    map_orig = Map(city_count)
    map_orig.generate_cities()

    map_tabu_1, fit_map_best, fit_map_immediate = tabu_search(map_orig,
                                                              max_tabu_size=100,
                                                              max_iter_count_multiplier=100,
                                                              neighbour_count=16,
                                                              max_non_improvement_count_multiplier=1.5)
    map_tabu_1.draw(append="tabu" + str(city_count))

    sns.set(rc={'figure.figsize': (10, 10)})
    ax = sns.lineplot(
        data=fit_map_best,
        errorbar=None, estimator=None, label="best", dashes=True)
    ax = sns.lineplot(
        data=fit_map_immediate,
        errorbar=None, estimator=None, label="immediate", dashes=True)
    ax.set(xlabel='Iterations', ylabel='Temperature')
    ax.set(yscale='log')
    plt.savefig("tabu_map.png")
    plt.clf()

    print(fit_map_best[-1])


def main():
    annealing_test(city_count=40)
    annealing_test(city_count=30)

    tabu_test(40)
    tabu_test(30)


if __name__ == '__main__':
    main()
