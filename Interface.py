import Maze
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import pygame
import os

def initialize_folders(algorithms):
    for alg in algorithms:
        if not os.path.exists(f"path/{alg.lower()}/"):
            os.makedirs(f"path/{alg.lower()}/")
        if not os.path.exists(f"visited/{alg.lower()}/"):
            os.makedirs(f"visited/{alg.lower()}/")
        Maze.delete_files_in_directory(f"path/{alg.lower()}/")
        Maze.delete_files_in_directory(f"visited/{alg.lower()}/")


def generate_maze():

    selected_algorithm = algorithms_var.get()
    maze_size = int(size_entry.get())
    maze = Maze.Maze(maze_size, 0.3)
    path = []
    if selected_algorithm == "BFS":
        path, visited = maze.bfs()
    elif selected_algorithm == "DFS":
        path, visited = maze.dfs()
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
        maze.render_visited(screen, window_width, window_height, visited[:i+1])
        view = pygame.surfarray.array3d(screen)

        img = Image.fromarray(view, 'RGB')
        img.save(f"visited/{selected_algorithm.lower()}/{selected_algorithm.lower()}_{count:03}.png")

    pygame.quit()
    display_slideshow(selected_algorithm)

def display_slideshow(selected_algorithm):
    image_folder = f"visited/{selected_algorithm.lower()}"
    image_list = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".png")]
    image_list.sort()

    def update_image(index=0):
        image = Image.open(image_list[index])
        image = image.resize((400, 400))
        img_tk = ImageTk.PhotoImage(image)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        index += 1
        if index < len(image_list):
            root.after(100, update_image, index)
    
    update_image()

initialize_folders(["BFS", "DFS", "A Star Manhattan", "A Star Diagonal", "A Star Euclidean"])

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
algorithms = ["BFS", "DFS", "A Star Manhattan", "A Star Diagonal", "A Star Euclidean"]
algorithms_var = tk.StringVar()
algorithms_var.set(algorithms[0])
algorithms_menu = tk.OptionMenu(left_frame, algorithms_var, *algorithms)
algorithms_menu.pack(pady=5)

# Entry field for maze size
size_label = tk.Label(left_frame, text="Enter Maze Size:")
size_label.pack(pady=5)
size_entry = tk.Entry(left_frame)
size_entry.pack(pady=5)

# Generate maze button
generate_button = tk.Button(left_frame, text="Generate Maze", command=generate_maze)
generate_button.pack(pady=5)

# Widget for the right frame
image_label = tk.Label(right_frame)
image_label.pack()

root.mainloop()