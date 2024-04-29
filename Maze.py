# RUN THESE COMMANDS IN TERMINAL
# pip install gymnasium pyvirtualdisplay
# python.exe -m pip install --upgrade pip --user
# pip install pygame

from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import heapq
from IPython.display import clear_output
from matplotlib import pyplot as plt
from PIL import Image

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
    
    # Checks if cell is within the bounds of the maze
    def is_valid(self, row, col):
        return (row >= 0) and (row < self.maze_length) and (col >= 0) and (col < self.maze_length)
    
    # Checks that cell is valid and open
    def is_open(self, row, col):
        if self.is_valid(row, col) == False:
            return False
        else:
            return self.maze[row][col] == 'O'
        
    # Checks if cell is valid and the goal
    def is_goal(self, row, col):
        if self.is_valid(row, col) == False:
            return False
        else:
            return self.maze[row][col] == 'G'
        
    def bfs(self):
        # Starting point
        start_row, start_col = 0, 0
        # Check if start is the goal
        if self.is_goal(start_row, start_col):
            return [(start_row, start_col)], [(start_row, start_col)]

        # Queue for BFS with (row, col, path taken to reach here)
        queue = deque([(start_row, start_col, [(start_row, start_col)])])
        visited = set()
        visited.add((start_row, start_col))

        while queue:
            current_row, current_col, path = queue.popleft()

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = current_row + dr, current_col + dc

                if self.is_valid(new_row, new_col) and (new_row, new_col) not in visited:
                    if self.is_open(new_row, new_col) or self.is_goal(new_row, new_col):
                        new_path = path + [(new_row, new_col)]
                        if self.is_goal(new_row, new_col):
                            return new_path, list(visited)
                        queue.append((new_row, new_col, new_path))
                        visited.add((new_row, new_col))

        return [], list(visited)

    # Returns a list for the solution and a list for the cells visited
    def a_star_manhattan(self):
        pass

    # Returns a list for the solution and a list for the cells visited
    def a_star_euclidean(self):
        pass

    def _render_frame(self, screen, window_width, window_height):
        screen.fill("white")

        cell_x = window_width / self.maze_length 
        cell_y = window_height / self.maze_length 

        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] == 'X':
                    pygame.draw.rect(screen, "black", pygame.Rect(cell_y * i, cell_x * j, cell_y, cell_x))
                elif self.maze[i][j] == 'G':
                    pygame.draw.rect(screen, "green", pygame.Rect(cell_y * i, cell_x * j, cell_y, cell_x))

    def render(self, screen, window_width, window_height):
        return self._render_frame(screen, window_width, window_height)

# TESTING CODE
maze = Maze(10)
print(maze.maze)
print(maze.manhattan_distance(0, 0, 3, 7))
print(maze.euclidean_distance(0, 0, 10, 10))


window_width = 400
window_height = 400
pygame.init()
screen = pygame.display.set_mode((window_width, window_height))
 

maze.render(screen, window_width, window_height)
view = pygame.surfarray.array3d(screen)

# displaying using plt
plt.imshow(view, interpolation='nearest')
plt.show()

# displaying using PIL, saves as local file
img = Image.fromarray(view, 'RGB')
#with open("my.png", 'wb') as f:
img.save("maze.png")
img.show()