# RUN THESE COMMANDS IN TERMINAL
# pip install gymnasium pyvirtualdisplay
# python.exe -m pip install --upgrade pip --user
# pip install pygame

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

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
    
    # Manhattan distance heuristic function
    def manhattan_distance(self, cur_row, cur_col, goal_row, goal_col):
        return abs(cur_row - goal_row) + abs(cur_col - goal_col)
    
    # Euclidean distance heuristic function
    def euclidean_distance(self, cur_row, cur_col, goal_row, goal_col):
        return math.sqrt((cur_row - goal_row)**2 + (cur_col - goal_col)**2)

# TESTING CODE
maze = Maze(10)
print(maze.maze)
print(maze.manhattan_distance(0, 0, 3, 7))
print(maze.euclidean_distance(0, 0, 10, 10))