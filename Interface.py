import Maze
from tkinter import *
from tkinter import ttk


window_height = 720
window_width = 1280
font_size = 12
window = Tk()
frame_ = ttk.Frame(window, padding=10)

"""
maze = Maze(10, 0.3)

window_width = 400
window_height = 400
pygame.init()
screen = pygame.display.set_mode((window_width, window_height))

path, visited = maze.dfs()
print(path)
print(visited)

if not os.path.exists("frames/dfs/"):
    os.makedirs("frames/dfs/")
if not os.path.exists("frames/bfs/"):
    os.makedirs("frames/bfs/")
count = 0
for (row, col) in path:

    count += 1
    maze._render_frame(screen, window_width, window_height, row, col)
    view = pygame.surfarray.array3d(screen)

    img = Image.fromarray(view, 'RGB')
    img.save(f"frames/dfs/dfs_{count}.png")
    # img.show()
    # time.sleep(1)

path, visited = maze.bfs()
print(path)
print(visited)

count = 0
for (row, col) in path:

    count += 1
    maze._render_frame(screen, window_width, window_height, row, col)
    view = pygame.surfarray.array3d(screen)

    img = Image.fromarray(view, 'RGB')
    img.save(f"frames/bfs/bfs_{count}.png")
    # img.show()
    # time.sleep(1)"""