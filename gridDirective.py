# import math
# stringsToParse = ("G6 V5R15", "G6W6 V4R7", "G6W2 V5R10")
# for s in stringsToParse:
#     gridSize = int(s[1:s.index(" ")])
#     if 'W' in s:
#         widthSeparator = s[s.index("W") + 1:]
#         gridWidth = int(s[0:s.index(" ")])
#     else:
#         smallLen = gridSize
#         currWidth = 0
#         for i in range(1, gridSize + 1):
#             if gridSize % i == 0:
#                 if abs(i - (int(gridSize/i))) < smallLen:
#                     smallLen = abs(i - (int(gridSize/i)))
#                     currWidth = i
        
#import sys; args = sys.argv[1:]

args = "4 G0 B0 B0E".split(" ")
import math

def get_neighbors(r, c, grid):
    neighbors = []
    if r > 0:
        neighbors.append((r - 1, c))
    if r < len(grid) - 1:
        neighbors.append((r + 1, c))
    if c > 0:
        neighbors.append((r, c - 1))
    if c < len(grid[0]) - 1:
        neighbors.append((r, c + 1))
    return neighbors

def get_best_neighbor(r, c, grid):
    neighbors = get_neighbors(r, c, grid)
    best_reward = -math.inf
    best_neighbor = None
    for nr, nc in neighbors:
        if grid[nr][nc][0] != "." and grid[nr][nc][0] != "*":
            reward = int(grid[nr][nc][0])
            if reward > best_reward:
                best_reward = reward
                best_neighbor = (nr, nc)
    return best_neighbor

# def get_reward(r, c, grid, val_func):
#     if grid[r][c][0] == "*":
#         return int(grid[r][c][1:])
#     if grid[r][c][0] == ".":
#         return -math.inf
#     if val_func == 0:
#         return int(grid[r][c][0])
#     if val_func == 1:
#         reward = int(grid[r][c][0])
#         best_neighbor = get_best_neighbor(r, c, grid)
#         if best_neighbor is None:
#             return -math.inf
#         steps = abs(best_neighbor[0] - r) + abs(best_neighbor[1] - c)
#         if steps == 0:
#             return reward
#         return reward / steps
def get_reward(r, c, grid, val_func):
    if not grid[r][c]:
        return -math.inf
    if grid[r][c][0] == "*":
        return int(grid[r][c][1:])
    if grid[r][c][0] == ".":
        return -math.inf
    if val_func == 0:
        return int(grid[r][c][0])
    if val_func == 1:
        reward = int(grid[r][c][0])
        best_neighbor = get_best_neighbor(r, c, grid)
        if best_neighbor is None:
            return -math.inf
        steps = abs(best_neighbor[0] - r) + abs(best_neighbor[1] - c)
        if steps == 0:
            return reward
        return reward / steps

def setup_gridworld(size, directives):
    width = int(math.sqrt(size))
    if size % width != 0:
        width += 1
    grid = [["" for _ in range(width)] for _ in range(width)]
    implied_reward = 12
    val_func = 1
    for directive in directives:
        if directive.startswith("R:"):
            implied_reward = int(directive[2:])
        elif directive.startswith("R"):
            pos, reward = directive.split(":")
            row, col = divmod(int(pos[1:]), width)
            grid[row][col] = "*" + reward
        elif directive.startswith("B"):
            if len(directive) == 2:
                row, col = divmod(int(directive[1:]), width)
                for neighbor in get_neighbors(row, col, grid):
                    grid[neighbor[0]][neighbor[1]] = "." + grid[neighbor[0]][neighbor[1]][1:]
                    grid[row][col] = grid[row][col][0] + "."
            else:
                pos = int(directive[1:2])
                row, col = divmod(pos, width)
                directions = directive[2:]
                for direction in directions:
                    if direction == "N" and row > 0:
                        grid[row][col] = grid[row][col][:-1] + "U"
                        grid[row-1][col] = grid[row-1][col][:-1] + "D"
                    elif direction == "S" and row < width - 1:
                        grid[row][col] = grid[row][col][:-1] + "D"
                        grid[row+1][col] = grid[row+1][col][:-1] + "U"
                    elif direction == "E" and col < width - 1:
                        grid[row][col] = grid[row][col][:-1] + "R"
                        grid[row][col+1] = grid[row][col+1][:-1] + "L"
                    elif direction == "W" and col > 0:
                        grid[row][col] = grid[row][col][:-1] + "L"
                        grid[row][col-1] = grid[row][col-1][:-1] + "R"
    # Generate the output grid
    output_grid = []
    for r in range(width):
        row = []
        for c in range(width):
            if grid[r][c].startswith("*"):
                row.append(int(grid[r][c][1:]))
            elif grid[r][c].startswith("."):
                row.append(-1)
            else:
                reward = get_reward(r, c, grid, val_func)
                row.append(reward)
        output_grid.append(row)
    return output_grid
print(setup_gridworld(int(args[0]), args[1:]))


# def get_reward(cell, rewards):
#     if cell in rewards:
#         return rewards[cell]
#     else:
#         return 0

# def generate_gridworld(directives):
#     size = int(directives[0])
#     width = int(math.sqrt(size))
#     if len(directives) > 1 and directives[1][1:].isdigit():
#         width = int(directives[1][1:])
#     rewards = {}
#     links = set()
#     implied_reward = 12
#     maximize_reward = False
#     for directive in directives[1:]:
#         if directive.startswith("R"):
#             if directive.count(":") == 1:
#                 pos, reward = map(int, directive[1:].split(":"))
#                 rewards[(pos // width, pos % width)] = reward
#             else:
#                 implied_reward = int(directive[1:])
#         elif directive.startswith("B"):
#             if directive[1:].isdigit():
#                 pos = int(directive[1:])
#                 links ^= {(pos, pos - width), (pos, pos + width), (pos, pos - 1), (pos, pos + 1)}
#             else:
#                 pos = int(directive[1:2])
#                 for d in directive[2:]:
#                     if d == "N":
#                         links ^= {(pos, pos - width), (pos - width, pos)}
#                     elif d == "S":
#                         links ^= {(pos, pos + width), (pos + width, pos)}
#                     elif d == "W":
#                         links ^= {(pos, pos - 1), (pos - 1, pos)}
#                     elif d == "E":
#                         links ^= {(pos, pos + 1), (pos + 1, pos)}
#         elif directive.startswith("G"):
#             maximize_reward = directive[1][1:] == "0"
#     grid = [["" for _ in range(width)] for _ in range(width)]
#     for r in range(width):
#         for c in range(width):
#             if (r, c) in rewards:
#                 grid[r][c] = "*"
#             else:
#                 possible_directions = ""
#                 reward_directions = {}
#                 for d in "URDL":
#                     next_r, next_c = r, c
#                     if d == "U":
#                         next_r -= 1
#                     elif d == "R":
#                         next_c += 1
#                     elif d == "D":
#                         next_r += 1
#                     elif d == "L":
#                         next_c -= 1
#                     if 0 <= next_r < width and 0 <= next_c < width and (r * width + c, next_r * width + next_c) in links:
#                         reward = get_reward((next_r, next_c), rewards)
#                         if maximize_reward:
#                             if reward in reward_directions:
#                                 reward_directions[reward] += d
#                             else:
#                                 reward_directions[reward] = d
#                         else:
#                             steps = abs(next_r - r) + abs(next_c - c)
#                             if reward != 0:
#                                 value = reward / steps
#                                 if value in reward_directions:
#                                     reward_directions[value] += d
#                                 else:
#                                     reward_directions[value] = d
#                 if not reward_directions:
#                     grid[r][c] = "."
#                 elif maximize_reward:
#                     grid[r][c] = reward_directions[max(reward_directions)]
#                 else:
#                     grid[r][c] = reward_directions[max(reward_directions.keys())]
#     return grid
# print(generate_gridworld(args))
# def create_gridworld(args):
#     # Parse input arguments
#     size = int(args[0])
#     width = int(math.sqrt(size)) if len(args) < 2 else int(args[1][1:])
#     print(width)
#     reward_type = "G1"
#     rewards = {}
#     links = {}
#     for i in range(size):
#         links[i] = set(["U", "R", "D", "L"])
#     implied_reward = 12
    
#     # Process remaining directives
#     for directive in args[2:]:
#         if directive[0] == "R":
#             if ":" in directive:
#                 pos, reward = directive[1:].split(":")
#                 rewards[int(pos)] = int(reward)
#             else:
#                 implied_reward = int(directive[1:])
#         elif directive[0] == "B":
#             if ":" in directive:
#                 pos, dirs = directive[1:].split(":")
#             else:
#                 pos = directive[1:]
#                 dirs = "UDLR"
#             for d in dirs:
#                 if d in links[int(pos)]:
#                     links[int(pos)].remove(d)
#                     if d == "U":
#                         links[int(pos)-width].remove("D")
#                     elif d == "R":
#                         links[int(pos)+1].remove("L")
#                     elif d == "D":
#                         links[int(pos)+width].remove("U")
#                     elif d == "L":
#                         links[int(pos)-1].remove("R")
#                 else:
#                     links[int(pos)].add(d)
#                     if d == "U":
#                         links[int(pos)-width].add("D")
#                     elif d == "R":
#                         links[int(pos)+1].add("L")
#                     elif d == "D":
#                         links[int(pos)+width].add("U")
#                     elif d == "L":
#                         links[int(pos)-1].add("R")
#         elif directive == "G0":
#             reward_type = "G0"
    
#     # Compute optimal directions for each cell
#     directions = {}
#     for i in range(size):
#         queue = [(i, "", 0)]
#         visited = set()
#         best_reward = -float("inf")
#         while queue:
#             pos, path, reward = queue.pop(0)
#             if pos in visited:
#                 continue
#             visited.add(pos)
#             if pos in rewards:
#                 if reward > best_reward:
#                     best_reward = reward
#                     directions[i] = path
#                 elif reward == best_reward:
#                     directions[i] += path
#             for d in links[pos]:
#                 if d == "U":
#                     new_pos = pos - width
#                 elif d == "R":
#                     new_pos = pos + 1
#                 elif d == "D":
#                     new_pos = pos + width
#                 elif d == "L":
#                     new_pos = pos - 1
#                 if new_pos >= 0 and new_pos < size and new_pos not in visited:
#                     if reward_type == "G0":
#                         new_reward = rewards.get(new_pos, -float("inf"))
#                         if new_reward >= best_reward:
#                             queue.append((new_pos, path+d, new_reward))
#                     else:
#                         new_reward = rewards.get(new_pos, 0)
#                         if reward + new_reward >= best_reward:
#                             queue.append((new_pos, path+d, reward+new_reward))
#         if i not in directions:
#             directions[i] = "."
    
#     # Construct output grid
#     grid = ["."] * size
#     print(rewards)
#     for i in range(size):
#         if i in rewards:
#             grid[i] = "*"
#         else:
#             grid[i] = directions[i]
#     print(grid)
# #print(args)
# #exit()
# create_gridworld(args)           
# #Arav Singh, Pd. 2, 2023