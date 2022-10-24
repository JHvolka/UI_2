import itertools
import math
import random
from dataclasses import dataclass, field
from PIL import Image, ImageDraw
import os
from copy import deepcopy


class PopulationSizeError(Exception):
    def __init__(self, size, message="Population size not correct"):
        self.size = size
        self.message = message
        super().__init__(self.message)


@dataclass
class City:
    x: float
    y: float
    identifier: int = field(default_factory=itertools.count().__next__, init=False)

    def __str__(self):
        return f"City {self.identifier}"

    def __repr__(self):
        return f"City {self.identifier}"


class Map:
    newid = itertools.count().__next__

    def __init__(self, number_of_cities: int):
        self.id = Map.newid()
        self.cities: list[City] = []
        self.number_of_cities = number_of_cities

    def generate_cities(self):
        self.cities = []
        for i in range(0, self.number_of_cities):
            self.cities.append(City(random.uniform(-200.0, 200.0), random.uniform(-200.0, 200.0)))

    def set_cities(self, city_list: list[City]):
        if len(city_list) != self.number_of_cities:
            raise PopulationSizeError
        self.cities = city_list

    def get_fitness(self):
        total = 0
        for idx, val in enumerate(self.cities):
            total = math.dist([self.cities[idx].x, self.cities[idx].y],
                              [self.cities[idx - 1].x, self.cities[idx - 1].y])
        return total

    def draw(self, path: str = "images/", generation: int = 0, number: int = 0):
        image = Image.new('RGB', (2100, 2100), (0, 0, 0))
        draw = ImageDraw.Draw(image)

        for idx, val in enumerate(self.cities):
            draw.line(((int(self.cities[idx].x) + 200) * 5 + 50, (int(self.cities[idx].y) + 200) * 5 + 50,
                       (int(self.cities[idx - 1].x) + 200) * 5 + 50, (int(self.cities[idx - 1].y) + 200) * 5 + 50),
                      fill=(0, 235, 180), width=1)
        for val in self.cities:
            draw.ellipse(((int(val.x) + 200) * 5 + 40, (int(val.y) + 200) * 5 + 40,
                          (int(val.x) + 200) * 5 + 60, (int(val.y) + 200) * 5 + 60),
                         outline=(0, 255, 200), fill=(0, 130, 80))

        # Create path to images if it does not exist
        exist = os.path.exists(path)
        if not exist:
            os.makedirs(path)

        #
        image.save(f'{path}map_{generation}_{number}.png')
        # with open(f'{path}map_{generation}_{number}_{self.id}.png', 'wb') as f:
        #     w = png.Writer(width, height, greyscale=False)
        #     w.write(f, img)
        return f'{path}map_{generation}_{number}_{self.id}.png'


class Population:
    def __init__(self, maps_number: int, cities_number: int):
        self.maps_number = maps_number
        self.cities_number = cities_number
        self.maps = [Map(cities_number) for x in range(0, maps_number)]


def generate_gif(self, path_list: list[str]):
    # TODO
    pass


def main():
    gen = Map(100)
    gen.generate_cities()
    print(gen.get_fitness())
    gen.draw()


if __name__ == '__main__':
    main()
