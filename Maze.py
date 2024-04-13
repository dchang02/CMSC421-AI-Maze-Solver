# RUN THESE COMMANDS IN TERMINAL
# pip install gymnasium pyvirtualdisplay
# python.exe -m pip install --upgrade pip --user
# pip install pygame

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class Maze():

    def __init__(self, maze_length) -> None:
        self.maze_length = maze_length
        self.maze = self.generate_maze(maze_length)

    # Generates maze by using 'X' as walls and 'O' as open space
    def generate_maze(self, n):
        symbols = ['X', 'O']
        wall_probability = 0.5  # Probability of a cell being a wall. Can change based on difficulty of maze
        open_probability = 1 - wall_probability
        matrix = np.random.choice(symbols, size=(n, n), p=[wall_probability, open_probability])
        matrix[0, 0] = 'O'  # Guarantees start is open
        matrix[-1, -1] = 'G'    # Makes the bottom-right cell the goal
        return matrix

# TESTING CODE
maze = Maze(10)
print(maze.maze)