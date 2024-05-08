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
import os

class Maze():

    def __init__(self, maze_length, wall_probability) -> None:
        self.maze_length = maze_length
        self.wall_probability = wall_probability
        self.maze = self.generate_maze(maze_length)

    # Generates maze by using 'X' as walls and 'O' as open space
    def generate_maze(self, n):
        symbols = ['X', 'O']
        open_probability = 1 - self.wall_probability
        matrix = np.random.choice(symbols, size=(n, n), p=[self.wall_probability, open_probability])
        matrix[0, 0] = 'O'  # Guarantees start is open
        matrix[-1, -1] = 'G'    # Makes the bottom-right cell the goal
        return matrix
    
    # Adjusts the probability of wall generation to change maze difficulty
    # Regerates a newly generated maze with the new difficulty
    def change_maze_difficulty(self, difficulty):

        if difficulty == 0: # Open maze
            self.wall_probability = 0.0
        elif difficulty == 1:   # Easy difficulty
            self.wall_probability = 0.1
        elif difficulty == 2:   # Medium difficulty
            self.wall_probability = 0.2
        elif difficulty == 3:   # Hard difficulty
            self.wall_probability = 0.35
        elif difficulty == 4:   # Expert difficulty
            self.wall_probability = 0.5

        # Returns the new maze
        self.maze = self.generate_maze(self.maze_length)

    # Adjust the size of the maze. Returns a newly generated maze with the new size
    def change_maze_size(self, n):
        
        if n < 4:   # Maze length must be at least 4
            print("The minimum maze length is 4")
        elif n > 100:   # Maze length must be at most 100
            print("The maximum maze length is 100")
        else:
            self.maze_length = n
            return self.generate_maze(n)    # Returns the new maze

    # Manhattan distance heuristic function
    def manhattan_distance(self, cur_row, cur_col, goal_row, goal_col):
        return abs(cur_row - goal_row) + abs(cur_col - goal_col)
    
    # Euclidean distance heuristic function
    def euclidean_distance(self, cur_row, cur_col, goal_row, goal_col):
        return math.sqrt((cur_row - goal_row)**2 + (cur_col - goal_col)**2)
    
    # Diagonal distance heuristic function
    def diagonal_distance(self, cur_row, cur_col, goal_row, goal_col):
        d_row = abs(cur_row - goal_row)
        d_col = abs(cur_col - goal_col)
        d = 1
        d2 = math.sqrt(2)
        return d * (d_row + d_col) + (d2 - 2 * d) * min(d_row, d_col)
    
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

        start_row, start_col = 0, 0

        if self.is_goal(start_row, start_col):
            return [(start_row, start_col)], [(start_row, start_col)]

        # Queue for BFS with (row, col, path taken to reach here)
        queue = deque([(start_row, start_col, [(start_row, start_col)])])
        visited = []
        visited.append((start_row, start_col))

        while queue:
            current_row, current_col, path = queue.popleft()

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = current_row + dr, current_col + dc

                if self.is_valid(new_row, new_col) and (new_row, new_col) not in visited:
                    if self.is_open(new_row, new_col) or self.is_goal(new_row, new_col):
                        new_path = path + [(new_row, new_col)]
                        if self.is_goal(new_row, new_col):
                            return new_path, visited
                        queue.append((new_row, new_col, new_path))
                        visited.append((new_row, new_col))

        return [], visited
    
    def dfs(self):
        start_row, start_col = 0, 0
        if self.is_goal(start_row, start_col):
            return [(start_row, start_col)], [(start_row, start_col)]

        # Stack for DFS with (row, col, path taken to reach here)
        stack = [(start_row, start_col, [(start_row, start_col)])]
        visited = []
        visited.append((start_row, start_col))

        while stack:
            current_row, current_col, path = stack.pop()

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = current_row + dr, current_col + dc

                if self.is_valid(new_row, new_col) and (new_row, new_col) not in visited:
                    if self.is_open(new_row, new_col) or self.is_goal(new_row, new_col):
                        new_path = path + [(new_row, new_col)]
                        if self.is_goal(new_row, new_col):
                            return new_path, visited
                        stack.append((new_row, new_col, new_path))
                        visited.append((new_row, new_col))

        return [], visited

    # Returns the path from the starting cell to the goal cell
    def create_path(self, start, previous_cells, current):
        
        path = []

        while current in previous_cells:
            path.append(current)
            current = previous_cells[current]

        path.append(start)
        path.reverse()

        return path

    # Runs the A* Search algorithm using the Manhattan distance heuristic.
    # Returns a list for the solution path and a list for the cells visited
    def a_star_manhattan(self):
        
        start = (0, 0)
        goal = (self.maze_length - 1, self.maze_length - 1)

        g_cost = {start: 0}
        queue = [(0, start)]
        searched_cells = []
        previous_cells = {}

        heapq.heappush(queue, (0, start))

        # Runs until there are no more cells to be checked
        while queue:

            _, current = heapq.heappop(queue)

            # Does not keep track of repeated visits to a cell
            if current not in searched_cells:
                searched_cells.append(current)

            # Creates solution path when goal is found
            if self.is_goal(current[0], current[1]):
                return self.create_path(start, previous_cells, current), searched_cells
            
            adjacent = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # Explores the neighbors of the current cell
            for row_diff, col_diff in adjacent:
                row = current[0] + row_diff
                col = current[1] + col_diff
                neighbor = (row, col)

                curr_g_cost = g_cost[current] + 1
                
                # Checks that the cell is valid and is not a wall
                if self.is_valid(row, col) and (self.is_open(row, col) or self.is_goal(row, col)):

                    # Updates g_cost if a better path is found
                    if neighbor not in g_cost or curr_g_cost < g_cost[neighbor]:
                        g_cost[neighbor] = curr_g_cost
                        f_cost = curr_g_cost + self.manhattan_distance(neighbor[0], neighbor[1], goal[0], goal[1])
                        previous_cells[neighbor] = current
                        heapq.heappush(queue, (f_cost, neighbor))
    
        # No path found
        return [], searched_cells

    # Runs the A* Search algorithm using the Euclidean distance heuristic.
    # Returns a list for the solution path and a list for the cells visited
    def a_star_euclidean(self):
        
        start = (0, 0)
        goal = (self.maze_length - 1, self.maze_length - 1)

        g_cost = {start: 0}
        queue = [(0, start)]
        searched_cells = []
        previous_cells = {}

        heapq.heappush(queue, (0, start))

        # Runs until there are no more cells to be checked
        while queue:

            _, current = heapq.heappop(queue)

            # Does not keep track of repeated visits to a cell
            if current not in searched_cells:
                searched_cells.append(current)

            # Creates solution path when goal is found
            if self.is_goal(current[0], current[1]):
                return self.create_path(start, previous_cells, current), searched_cells
            
            adjacent = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # Explores the neighbors of the current cell
            for row_diff, col_diff in adjacent:
                row = current[0] + row_diff
                col = current[1] + col_diff
                neighbor = (row, col)

                curr_g_cost = g_cost[current] + 1
                
                # Checks that the cell is valid and is not a wall
                if self.is_valid(row, col) and (self.is_open(row, col) or self.is_goal(row, col)):

                    # Updates g_cost if a better path is found
                    if neighbor not in g_cost or curr_g_cost < g_cost[neighbor]:
                        g_cost[neighbor] = curr_g_cost
                        f_cost = curr_g_cost + self.euclidean_distance(neighbor[0], neighbor[1], goal[0], goal[1])
                        previous_cells[neighbor] = current
                        heapq.heappush(queue, (f_cost, neighbor))
    
        # No path found
        return [], searched_cells
    
    # Runs the A* Search algorithm using the Diagonal distance heuristic.
    # Returns a list for the solution path and a list for the cells visited
    def a_star_diagonal(self):
        
        start = (0, 0)
        goal = (self.maze_length - 1, self.maze_length - 1)

        g_cost = {start: 0}
        queue = [(0, start)]
        searched_cells = []
        previous_cells = {}

        heapq.heappush(queue, (0, start))

        # Runs until there are no more cells to be checked
        while queue:

            _, current = heapq.heappop(queue)

            # Does not keep track of repeated visits to a cell
            if current not in searched_cells:
                searched_cells.append(current)

            # Creates solution path when goal is found
            if self.is_goal(current[0], current[1]):
                return self.create_path(start, previous_cells, current), searched_cells
            
            adjacent = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # Explores the neighbors of the current cell
            for row_diff, col_diff in adjacent:
                row = current[0] + row_diff
                col = current[1] + col_diff
                neighbor = (row, col)

                curr_g_cost = g_cost[current] + 1
                
                # Checks that the cell is valid and is not a wall
                if self.is_valid(row, col) and (self.is_open(row, col) or self.is_goal(row, col)):

                    # Updates g_cost if a better path is found
                    if neighbor not in g_cost or curr_g_cost < g_cost[neighbor]:
                        g_cost[neighbor] = curr_g_cost
                        f_cost = curr_g_cost + self.diagonal_distance(neighbor[0], neighbor[1], goal[0], goal[1])
                        previous_cells[neighbor] = current
                        heapq.heappush(queue, (f_cost, neighbor))
    
        # No path found
        return [], searched_cells

    # Runs the Greedy Search algorithm using the Euclidean distance heuristic.
    # Returns a list for the solution path and a list for the cells visited
    def greedy_euclidean(self):
        start_state = (0,0)
        goal_state = (self.maze_length - 1, self.maze_length - 1)
        
        # visited set
        visited = set()
    
        # solution path    
        solution = []

        # states expanded
        explored = []

        predecessor_matrix = np.full((self.maze_length, self.maze_length), None, dtype=object)
        frontier = []

        # push start state into frontier
        item_to_push = (None, start_state)
        h_cost = self.euclidean_distance(start_state[0], start_state[1], goal_state[0], goal_state[1])
        heapq.heappush(frontier, (h_cost, item_to_push))

        # keep running until the frontier become empty
        while frontier:
            _, popped_item = heapq.heappop(frontier)
            came_from = popped_item[0]
            current_state = popped_item[1]

            # goal check
            if (current_state == goal_state):
                explored.append(current_state)
                predecessor_matrix[current_state[0], current_state[1]] = came_from

                # construct solution path
                solution.append(current_state)
                prev_state = predecessor_matrix[current_state[0], current_state[1]]
                while (prev_state != None):
                    solution.insert(0, prev_state)
                    prev_state = predecessor_matrix[prev_state[0], prev_state[1]]
                
                return solution, explored

            # do not expand a state that has already been expanded
            if (current_state not in visited):
                visited.add(current_state)
                explored.append(current_state)
                predecessor_matrix[current_state[0], current_state[1]] = came_from

                # push valid neigbors into frontier
                current_state_row = current_state[0]
                current_state_col = current_state[1]
                neighbors = [(current_state_row, current_state_col + 1), (current_state_row, current_state_col - 1), (current_state_row + 1, current_state_col), (current_state_row - 1, current_state_col)]
                for neighbor in neighbors:
                    neighbor_row = neighbor[0]
                    neighbor_col = neighbor[1]
                    
                    if (self.is_open(neighbor_row, neighbor_col) or self.is_goal(neighbor_row, neighbor_col)):
                        item_to_push = (current_state, neighbor)
                        h_cost = self.euclidean_distance(neighbor_row, neighbor_col, goal_state[0], goal_state[1])
                        heapq.heappush(frontier, (h_cost, item_to_push))
        
        # no path from start state to goal state
        return [], explored

    # Runs the Greedy Search algorithm using the Manhattan distance heuristic.
    # Returns a list for the solution path and a list for the cells visited
    def greedy_manhattan(self):
        start_state = (0,0)
        goal_state = (self.maze_length - 1, self.maze_length - 1)

        # visited set
        visited = set()
    
        # solution path    
        solution = []

        # states expanded
        explored = []

        predecessor_matrix = np.full((self.maze_length, self.maze_length), None, dtype=object)
        frontier = []

        # push start state into frontier
        item_to_push = (None, start_state)
        h_cost = self.manhattan_distance(start_state[0], start_state[1], goal_state[0], goal_state[1])
        heapq.heappush(frontier, (h_cost, item_to_push))

        # keep running until the frontier become empty
        while frontier:
            _, popped_item = heapq.heappop(frontier)
            came_from = popped_item[0]
            current_state = popped_item[1]

            # goal check
            if (current_state == goal_state):
                explored.append(current_state)
                predecessor_matrix[current_state[0], current_state[1]] = came_from

                # construct solution path
                solution.append(current_state)
                prev_state = predecessor_matrix[current_state[0], current_state[1]]
                while (prev_state != None):
                    solution.insert(0, prev_state)
                    prev_state = predecessor_matrix[prev_state[0], prev_state[1]]
                
                return solution, explored

            # do not expand a state that has already been expanded
            if (current_state not in visited):
                visited.add(current_state)
                explored.append(current_state)
                predecessor_matrix[current_state[0], current_state[1]] = came_from

                # push valid neigbors into frontier
                current_state_row = current_state[0]
                current_state_col = current_state[1]
                neighbors = [(current_state_row, current_state_col + 1), (current_state_row, current_state_col - 1), (current_state_row + 1, current_state_col), (current_state_row - 1, current_state_col)]
                for neighbor in neighbors:
                    neighbor_row = neighbor[0]
                    neighbor_col = neighbor[1]
                    
                    if (self.is_open(neighbor_row, neighbor_col) or self.is_goal(neighbor_row, neighbor_col)):
                        item_to_push = (current_state, neighbor)
                        h_cost = self.manhattan_distance(neighbor_row, neighbor_col, goal_state[0], goal_state[1])
                        heapq.heappush(frontier, (h_cost, item_to_push))
        
        # no path from start state to goal state
        return [], explored

    # Runs the Greedy Search algorithm using the Diagonal distance heuristic.
    # Returns a list for the solution path and a list for the cells visited
    def greedy_diagonal(self):
        start_state = (0,0)
        goal_state = (self.maze_length - 1, self.maze_length - 1)

        # visited set
        visited = set()
    
        # solution path    
        solution = []

        # states expanded
        explored = []

        predecessor_matrix = np.full((self.maze_length, self.maze_length), None, dtype=object)
        frontier = []

        # push start state into frontier
        item_to_push = (None, start_state)
        h_cost = self.diagonal_distance(start_state[0], start_state[1], goal_state[0], goal_state[1])
        heapq.heappush(frontier, (h_cost, item_to_push))

        # keep running until the frontier become empty
        while frontier:
            _, popped_item = heapq.heappop(frontier)
            came_from = popped_item[0]
            current_state = popped_item[1]

            # goal check
            if (current_state == goal_state):
                explored.append(current_state)
                predecessor_matrix[current_state[0], current_state[1]] = came_from

                # construct solution path
                solution.append(current_state)
                prev_state = predecessor_matrix[current_state[0], current_state[1]]
                while (prev_state != None):
                    solution.insert(0, prev_state)
                    prev_state = predecessor_matrix[prev_state[0], prev_state[1]]
                
                return solution, explored

            # do not expand a state that has already been expanded
            if (current_state not in visited):
                visited.add(current_state)
                explored.append(current_state)
                predecessor_matrix[current_state[0], current_state[1]] = came_from

                # push valid neigbors into frontier
                current_state_row = current_state[0]
                current_state_col = current_state[1]
                neighbors = [(current_state_row, current_state_col + 1), (current_state_row, current_state_col - 1), (current_state_row + 1, current_state_col), (current_state_row - 1, current_state_col)]
                for neighbor in neighbors:
                    neighbor_row = neighbor[0]
                    neighbor_col = neighbor[1]
                    
                    if (self.is_open(neighbor_row, neighbor_col) or self.is_goal(neighbor_row, neighbor_col)):
                        item_to_push = (current_state, neighbor)
                        h_cost = self.diagonal_distance(neighbor_row, neighbor_col, goal_state[0], goal_state[1])
                        heapq.heappush(frontier, (h_cost, item_to_push))
        
        # no path from start state to goal state
        return [], explored

    def render_agent(self, screen, window_width, window_height, row, col, visited, path):
        screen.fill("white")

        self.render_visited(screen, window_width, window_height, visited)
        cell_x = window_width / self.maze_length 
        cell_y = window_height / self.maze_length 

        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] == 'X':
                    pygame.draw.rect(screen, "black", pygame.Rect(cell_y * i, cell_x * j, cell_y, cell_x))
                elif self.maze[i][j] == 'G':
                    pygame.draw.rect(screen, "green", pygame.Rect(cell_y * i, cell_x * j, cell_y, cell_x))
        radius = min(cell_x / 2, cell_y / 2)
        circle_x = cell_x * row + radius
        circle_y = cell_y * col + radius

        circle_pos = pygame.Vector2(circle_x, circle_y)

        for (row, col) in path:
            pygame.draw.rect(screen, "green", pygame.Rect(cell_y * row, cell_x * col, cell_y, cell_x))

        pygame.draw.circle(screen, "blue", circle_pos, radius)

    def render_visited(self, screen, window_width, window_height, visited):
        screen.fill("white")

        cell_x = window_width / self.maze_length 
        cell_y = window_height / self.maze_length 

        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] == 'X':
                    pygame.draw.rect(screen, "black", pygame.Rect(cell_y * i, cell_x * j, cell_y, cell_x))
                elif self.maze[i][j] == 'G':
                    pygame.draw.rect(screen, "green", pygame.Rect(cell_y * i, cell_x * j, cell_y, cell_x))
        
        for (row, col) in visited:
            pygame.draw.rect(screen, "gray", pygame.Rect(cell_y * row, cell_x * col, cell_y, cell_x))

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
   except OSError:
     print("Error occurred while deleting files.")

# TESTING CODE
"""
# Maze(size, wall_prob)
maze = Maze(10, 0.3)


# initiating the canvas
window_width = 400
window_height = 400
pygame.init()
screen = pygame.display.set_mode((window_width, window_height))

# getting the path and visited from selected alg
# both variables are lists
path, visited = maze.bfs()
print(path)
print(visited)

# make the directory for storing images 
if not os.path.exists("path/bfs/"):
    os.makedirs("path/bfs/")
if not os.path.exists("visited/bfs/"):
    os.makedirs("visited/bfs/")
delete_files_in_directory("path/bfs/")
delete_files_in_directory("visited/bfs/")

count = 0

# for each pair of coords in solution path, render the frame and save it
for (row, col) in path:

    count += 1
    maze.render_agent(screen, window_width, window_height, row, col, [])
    view = pygame.surfarray.array3d(screen)

    img = Image.fromarray(view, 'RGB')
    img.save(f"path/bfs/bfs_{count}.png")
    # img.show()
    # time.sleep(1)

count = 0
# alternatively, for visited cells so far, a different render method is used
for i in range(len(visited)):

    count += 1
    maze.render_visited(screen, window_width, window_height, visited[:i+1])
    view = pygame.surfarray.array3d(screen)

    img = Image.fromarray(view, 'RGB')
    img.save(f"visited/bfs/bfs_{count}.png")
    # img.show()
    # time.sleep(1)"""