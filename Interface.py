import Maze
import tkinter as tk
from PIL import ImageTk, Image
import pygame
import os
import time


def initialize_folders(algorithms):
    for alg in algorithms:
        #if not os.path.exists(f"path/{alg.lower()}/"):
        #    os.makedirs(f"path/{alg.lower()}/")
        if not os.path.exists(f"visited/{alg.lower()}/"):
            os.makedirs(f"visited/{alg.lower()}/")
        #Maze.delete_files_in_directory(f"path/{alg.lower()}/")
        Maze.delete_files_in_directory(f"visited/{alg.lower()}/")

def reset_maze():

    global resetted
    image = Image.open("maze.png")
    image = image.resize((400, 400))
    img_tk = ImageTk.PhotoImage(image)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    result_label.config(text = "")
    resetted = True



def generate_maze():
    global resetted
    global maze
    resetted = True
    initialize_folders(["BFS", "DFS", "Greedy Manhattan", "Greedy Diagonal", "Greedy Euclidean", "A Star Manhattan", "A Star Diagonal", "A Star Euclidean"])
    result_label.config(text = "")
    maze_size = int(size_entry.get())
    maze_difficulty = int(difficulties_var.get())
    maze = Maze.Maze(maze_size, 0.3)
    maze.change_maze_difficulty(maze_difficulty)
    window_width = 400
    window_height = 400
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    maze.render_agent(screen, window_width, window_height, 0, 0, [])
    view = pygame.surfarray.array3d(screen)

    img = Image.fromarray(view, 'RGB')
    img.save(f"maze.png")
    
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    pygame.quit()
    

def solve_maze():

    global resetted
    if not resetted:
        return
    reset_maze()
    resetted = False
    initialize_folders(["BFS", "DFS", "Greedy Manhattan", "Greedy Diagonal", "Greedy Euclidean", "A Star Manhattan", "A Star Diagonal", "A Star Euclidean"])
    selected_algorithm = algorithms_var.get()
    path = []
    start_time = time.time()
    if selected_algorithm == "BFS":
        path, visited = maze.bfs()
    elif selected_algorithm == "DFS":
        path, visited = maze.dfs()
    elif selected_algorithm == "Greedy Manhattan":
        path, visited = maze.greedy_manhattan()
    elif selected_algorithm == "Greedy Diagonal":
        path, visited = maze.greedy_diagonal()
    elif selected_algorithm == "Greedy Euclidean":
        path, visited = maze.greedy_euclidean()    
    elif selected_algorithm == "A Star Manhattan":
        path, visited = maze.a_star_manhattan()
    elif selected_algorithm == "A Star Diagonal":
        path, visited = maze.a_star_diagonal()
    elif selected_algorithm == "A Star Euclidean":
        path, visited = maze.a_star_euclidean()
    print(visited)
    
    window_width = 400
    window_height = 400
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    count = 0
    for i in range(len(visited)):

        count += 1
        maze.render_agent(screen, window_width, window_height, 0, 0, visited[:i+1])
        view = pygame.surfarray.array3d(screen)

        img = Image.fromarray(view, 'RGB')
        img.save(f"visited/{selected_algorithm.lower()}/{selected_algorithm.lower()}_{count:03}.png")

    #count = 0
    for (row, col) in path:

        count += 1
        maze.render_agent(screen, window_width, window_height, row, col, visited)
        view = pygame.surfarray.array3d(screen)

        img = Image.fromarray(view, 'RGB')
        img.save(f"visited/{selected_algorithm.lower()}/{selected_algorithm.lower()}_{count:03}.png")

    pygame.quit()
    total_time = time.time() - start_time
    result_label.config(text = f"Time taken: {total_time} seconds\nVisited cells: {len(visited)}\nSolution path length: {len(path)}")
    display_slideshow(selected_algorithm)


def display_slideshow(selected_algorithm):

    image_folder = f"visited/{selected_algorithm.lower()}"
    image_list = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".png")]
    image_list.sort()
    
    index = 0 
    def update_image(index=0):
        global resetted
        if not resetted:
            image = Image.open(image_list[index])
            image = image.resize((400, 400))
            img_tk = ImageTk.PhotoImage(image)
            image_label.config(image=img_tk)
            image_label.image = img_tk
            index += 1
            if index < len(image_list) and not resetted:
                root.after(100, update_image, index)
    
    update_image(index)

initialize_folders(["BFS", "DFS", "Greedy Manhattan", "Greedy Diagonal", "Greedy Euclidean", "A Star Manhattan", "A Star Diagonal", "A Star Euclidean"])


maze = Maze.Maze(10, 0)
resetted = True
# Create main window
root = tk.Tk()
root.title("Maze Generator App")
root.geometry("800x400")

# Create left and right frames
left_frame = tk.Frame(root, width=400, height=400, bg='gray')
left_frame.pack_propagate(0)
left_frame.pack(side=tk.LEFT)

right_frame = tk.Frame(root, width=400, height=400, bg='white')
right_frame.pack_propagate(0)
right_frame.pack(side=tk.RIGHT)


# Dropdown menu for selecting algorithms
algorithms_label = tk.Label(left_frame, text="Select Algorithm:")
algorithms_label.pack(pady=5)

algorithms = ["BFS", "DFS", "Greedy Manhattan", "Greedy Diagonal", "Greedy Euclidean", "A Star Manhattan", "A Star Diagonal", "A Star Euclidean"]
algorithms_var = tk.StringVar()
algorithms_var.set(algorithms[0])
algorithms_menu = tk.OptionMenu(left_frame, algorithms_var, *algorithms)
algorithms_menu.pack(pady=5)

difficulties_label = tk.Label(left_frame, text="Select Difficulty:")
difficulties_label.pack(pady=5)

difficulties = ["0", "1", "2", "3", "4"]
difficulties_var = tk.StringVar()
difficulties_var.set(difficulties[1])
difficulties_menu = tk.OptionMenu(left_frame, difficulties_var, *difficulties)
difficulties_menu.pack(pady=5)

# Entry field for maze size
size_label = tk.Label(left_frame, text="Enter Maze Size:")
size_label.pack(pady=5)
size_entry = tk.Entry(left_frame)
size_entry.pack(pady=5)

# Generate maze button
generate_button = tk.Button(left_frame, text="Generate Maze", command=generate_maze)
generate_button.pack(pady=5)

solve_button = tk.Button(left_frame, text="Solve Maze", command=solve_maze)
solve_button.pack(pady=5)

reset_button = tk.Button(left_frame, text="Reset Maze", command=reset_maze)
reset_button.pack(pady=5)

result_label = tk.Label(left_frame, text="")
result_label.pack(pady=5)

# Widget for the right frame
image_label = tk.Label(right_frame)
image_label.pack()




root.mainloop()