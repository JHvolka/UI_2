import itertools
import math
import random
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
import os
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import time
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool


class PopulationSizeError(Exception):
    def __init__(self, size, message="Population size not correct"):
        self.size = size
        self.message = message
        super().__init__(self.message)


@dataclass
class City:
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
        self.id = Map.newid()
        self.cities: list[City] = []
        self.number_of_cities = number_of_cities
        self.fitness = None

    def get_num_of_cities(self):
        return self.number_of_cities

    def generate_cities(self):
        self.cities = []
        for i in range(0, self.number_of_cities):
            self.cities.append(City(random.uniform(-200.0, 200.0), random.uniform(-200.0, 200.0), i))

    def set_cities(self, city_list: list[City]):
        if len(city_list) != self.number_of_cities:
            raise PopulationSizeError
        self.cities = city_list

    def get_fitness(self):
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
             font_size: int = 20
             ):
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
        m = deepcopy(self)

        c_1 = int(random.uniform(0, self.number_of_cities))
        c_2 = int(random.uniform(0, self.number_of_cities))
        # c_1 = 2
        # c_2 = 1

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


def probability_function(len_1, len_2, temp):
    if len_1 == len_2:
        return 0
    elif len_1 > len_2:
        return 1
    try:
        return math.exp((len_1 - len_2) / temp)
    except OverflowError:
        print("AAAAAA")
        return 1


class Schedules(Enum):
    EXPONENTIAL = 1
    QUADRATIC = 2
    LINEAR = 3


def simulated_annealing(m: Map, max_states_multiplier: int = 30,
                        neighbour_count_multiplier: float = 2,
                        initial_temp_multiplier: float = 500,
                        cooling_factor: float = 0.99,
                        schedule: Schedules = Schedules.EXPONENTIAL):
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

    simulated.draw(append=name)
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
                                                               cooling_factor=0.9,
                                                               initial_temp_multiplier=1000)

    df, temp_map_lin, fit_map_lin = test_simulated_annealing(map_orig,
                                                             df,
                                                             "Linear",
                                                             schedule=Schedules.LINEAR,
                                                             initial_temp_multiplier=300)

    df, temp_map_exp, fit_map_exp = test_simulated_annealing(map_orig,
                                                             df,
                                                             "Exponential",
                                                             initial_temp_multiplier=500,
                                                             cooling_factor=0.99)

    if output_images:
        sns.set(rc={'figure.figsize': (10, 10)})
        ax = sns.lineplot(data=temp_map_quad, errorbar=None, estimator=None, label="quadratic")
        ax = sns.lineplot(data=temp_map_lin, errorbar=None, estimator=None, label="linear")
        ax = sns.lineplot(data=temp_map_exp, errorbar=None, estimator=None, label="exponential")
        ax.set(xlabel='Iterations', ylabel='Temperature')
        ax.set(yscale='log')
        plt.savefig("temp_map.png")
        plt.clf()
        ax = sns.lineplot(data=fit_map_quad, errorbar=None, estimator=None, label="quadratic")
        ax = sns.lineplot(data=fit_map_lin, errorbar=None, estimator=None, label="linear")
        ax = sns.lineplot(data=fit_map_exp, errorbar=None, estimator=None, label="exponential")
        ax.set(xlabel='Iterations', ylabel='Fitness')
        plt.savefig("fit_map.png")

    if print_table:
        print(df)
    return df


def main():
    annealing_test()


if __name__ == '__main__':
    main()
