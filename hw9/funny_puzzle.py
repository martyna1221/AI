import numpy as np
import math
import heapq

# given a state of the puzzle, represented as a single list of integers with
# a 0 in the empty space, print to the console all of the possible successor
# states
def print_succ(state):
  l = []
  i = state.index(0)
  if i == 0: # index 1 and 3 can be successors
    temp = list(state)
    index_one = temp[1]
    temp[1] = 0
    temp[0] = index_one

    l.append(temp)

    temp = list(state)
    index_three = temp[3]
    temp[3] = 0
    temp[0] = index_three

    l.append(temp)

    l = sorted(l)
    for e in l:
      print(e),
      print("h=%d" % Manhattan_distance(e))
    return
  if i == 1: # index 0, 2 and 4 can be successors
    temp = list(state)
    index_zero = temp[0]
    temp[0] = 0
    temp[1] = index_zero

    l.append(temp)

    temp = list(state)
    index_two = temp[2]
    temp[2] = 0
    temp[1] = index_two

    l.append(temp)

    temp = list(state)
    index_four = temp[4]
    temp[4] = 0
    temp[1] = index_four

    l.append(temp)

    l = sorted(l)
    for e in l:
      print(e),
      print("h=%d" % Manhattan_distance(e))
    return
  if i == 2: # index 1 and 5 can be successors
    temp = list(state)
    index_one = temp[1]
    temp[1] = 0
    temp[2] = index_one

    l.append(temp)

    temp = list(state)
    index_five = temp[5]
    temp[5] = 0
    temp[2] = index_five

    l.append(temp)

    l = sorted(l)
    for e in l:
      print(e),
      print("h=%d" % Manhattan_distance(e))
    return
  if i == 3: # index 0, 4 and 6 can be successors
    temp = list(state)
    index_zero = temp[0]
    temp[0] = 0
    temp[3] = index_zero

    l.append(temp)

    temp = list(state)
    index_four = temp[4]
    temp[4] = 0
    temp[3] = index_four

    l.append(temp)

    temp = list(state)
    index_six = temp[6]
    temp[6] = 0
    temp[3] = index_six

    l.append(temp)

    l = sorted(l)
    for e in l:
      print(e),
      print("h=%d" % Manhattan_distance(e))
    return
  if i == 4: # index 1, 3, 5 and 7 can be successors
    temp = list(state)
    index_one = temp[1]
    temp[1] = 0
    temp[4] = index_one

    l.append(temp)

    temp = list(state)
    index_three = temp[3]
    temp[3] = 0
    temp[4] = index_three

    l.append(temp)

    temp = list(state)
    index_seven = temp[7]
    temp[7] = 0
    temp[4] = index_seven

    l.append(temp)

    temp = list(state)
    index_five = temp[5]
    temp[5] = 0
    temp[4] = index_five

    l.append(temp)

    l = sorted(l)
    for e in l:
      print(e),
      print("h=%d" % Manhattan_distance(e))
    return
  if i == 5: # index 2, 4 and 8 can be successors
    temp = list(state)
    index_two = temp[2]
    temp[2] = 0
    temp[5] = index_two

    l.append(temp)

    temp = list(state)
    index_four = temp[4]
    temp[4] = 0
    temp[5] = index_four

    l.append(temp)

    temp = list(state)
    index_eight = temp[8]
    temp[8] = 0
    temp[5] = index_eight

    l.append(temp)

    l = sorted(l)
    for e in l:
      print(e),
      print("h=%d" % Manhattan_distance(e))
    return
  if i == 6: # index 3 and 7 can be successors
    temp = list(state)
    index_three = temp[3]
    temp[3] = 0
    temp[6] = index_three

    l.append(temp)

    temp = list(state)
    index_seven = temp[7]
    temp[7] = 0
    temp[6] = index_seven

    l.append(temp)

    l = sorted(l)
    for e in l:
      print(e),
      print("h=%d" % Manhattan_distance(e))
    return
  if i == 7: # index 4, 6 and 8 can be successors
    temp = list(state)
    index_four = temp[4]
    temp[4] = 0
    temp[7] = index_four

    l.append(temp)

    temp = list(state)
    index_six = temp[6]
    temp[6] = 0
    temp[7] = index_six

    l.append(temp)

    temp = list(state)
    index_eight = temp[8]
    temp[8] = 0
    temp[7] = index_eight

    l.append(temp)

    l = sorted(l)
    for e in l:
      print(e),
      print("h=%d" % Manhattan_distance(e))
    return
  if i == 8: # index 5 and 7 can be successors
    temp = list(state)
    index_five = temp[5]
    temp[5] = 0
    temp[8] = index_five

    l.append(temp)

    temp = list(state)
    index_seven = temp[7]
    temp[7] = 0
    temp[8] = index_seven

    l.append(temp)

    l = sorted(l)
    for e in l:
      print(e),
      print("h=%d" % Manhattan_distance(e))
    return

# given a state of the puzzle, perform the A* search algorithm and print the
# path from the current state to the goal state
def solve(state):
  pq = [] # my priority queue
  closed = {} # contains all popped pq entries
  l = [] # list that contains states of popped pq entries

  path = [] # list which stores the solution path

  g = 0 # number of moves
  p_index = -1 # starting parent index (root doesn't have a parent hence p_index = -1 -> invalid)
  h = Manhattan_distance(state) # h

  heapq.heappush(pq, (g+h, state, (g, h, p_index)))

  queue = True
  
  while queue: # while queue not empty or while queue hasn't reached an end condition
    b = heapq.heappop(pq) # pops lowest priority element
    p_index += 1 # increments parent index by 1
    closed[p_index] = b # adds popped element to dictionary 'closed'
    l.append(b[1]) # adds state of popped element to list 'l'
    g = b[2][0] + 1 # increments g (i.e. # of moves) accordingly

    if Manhattan_distance(b[1]) == 0: # if popped element has the correct state ...
      path.append(b) # appends solution entry
      index = b[2][2] # gets parent index of last element where h = 0

      while index != -1: # back trace to find 
        path.append(closed[index]) # adds state to list 'path'
        index = closed[index][2][2] # increments index

      queue = False
    s = get_successors(b[1]) # calls the get_successors() method on popped pq element
    for e in s:
      if e not in l: # if said path is NOT in visited list 'l'
        h = Manhattan_distance(e)
        heapq.heappush(pq, (g+h, e, (g, h, p_index)))

  for e in path[::-1]: # traverses list 'path' in reverse order
    print(e[1]),
    print("h=%d" % Manhattan_distance(e[1])), 
    print("moves: %d" % e[2][0])

  return

# given a state of the puzzle, 
def Manhattan_distance(state):
  counter = 0
  x = np.reshape(state, (3, 3))
  counter = 0

  c_1 = np.argwhere(x == 1) # 1 co-ordinate should be (0, 0)
  c_1_x = c_1[0][0]
  c_1_y = c_1[0][1]
  d = abs(0 - c_1_x) + abs(0 - c_1_y)
  counter += d

  c_2 = np.argwhere(x == 2) # 2 co-ordinate should be (0, 1)
  c_2_x = c_2[0][0]
  c_2_y = c_2[0][1]
  d = abs(0 - c_2_x) + abs(1 - c_2_y)
  counter += d

  c_3 = np.argwhere(x == 3) # 3 co-ordinate should be (0, 2)
  c_3_x = c_3[0][0]
  c_3_y = c_3[0][1]
  d = abs(0 - c_3_x) + abs(2 - c_3_y)
  counter += d

  c_4 = np.argwhere(x == 4) # 4 co-ordinate should be (1, 0)
  c_4_x = c_4[0][0]
  c_4_y = c_4[0][1]
  d = abs(1 - c_4_x) + abs(0 - c_4_y)
  counter += d

  c_5 = np.argwhere(x == 5) # 5 co-ordinate should be (1, 1)
  c_5_x = c_5[0][0]
  c_5_y = c_5[0][1]
  d = abs(1 - c_5_x) + abs(1 - c_5_y)
  counter += d

  c_6 = np.argwhere(x == 6) # 6 co-ordinate should be (1, 2)
  c_6_x = c_6[0][0]
  c_6_y = c_6[0][1]
  d = abs(1 - c_6_x) + abs(2 - c_6_y)
  counter += d

  c_7 = np.argwhere(x == 7) # 7 co-ordinate should be (2, 0)
  c_7_x = c_7[0][0]
  c_7_y = c_7[0][1]
  d = abs(2 - c_7_x) + abs(0 - c_7_y)
  counter += d

  c_8 = np.argwhere(x == 8) # 8 co-ordinate should be (2, 1)
  c_8_x = c_8[0][0]
  c_8_y = c_8[0][1]
  d = abs(2 - c_8_x) + abs(1 - c_8_y)
  counter += d
  return counter

# gets sucessors and returns sucessors in a sorted list given a given state
def get_successors(state):
  l = []
  i = state.index(0)
  if i == 0: # index 1 and 3 can be successors
    temp = list(state)
    index_one = temp[1]
    temp[1] = 0
    temp[0] = index_one

    l.append(temp)

    temp = list(state)
    index_three = temp[3]
    temp[3] = 0
    temp[0] = index_three

    l.append(temp)

    l = sorted(l)
    return l
  if i == 1: # index 0, 2 and 4 can be successors
    temp = list(state)
    index_zero = temp[0]
    temp[0] = 0
    temp[1] = index_zero

    l.append(temp)

    temp = list(state)
    index_two = temp[2]
    temp[2] = 0
    temp[1] = index_two

    l.append(temp)

    temp = list(state)
    index_four = temp[4]
    temp[4] = 0
    temp[1] = index_four

    l.append(temp)

    l = sorted(l)
    return l
  if i == 2: # index 1 and 5 can be successors
    temp = list(state)
    index_one = temp[1]
    temp[1] = 0
    temp[2] = index_one

    l.append(temp)

    temp = list(state)
    index_five = temp[5]
    temp[5] = 0
    temp[2] = index_five

    l.append(temp)

    l = sorted(l)
    return l
  if i == 3: # index 0, 4 and 6 can be successors
    temp = list(state)
    index_zero = temp[0]
    temp[0] = 0
    temp[3] = index_zero

    l.append(temp)

    temp = list(state)
    index_four = temp[4]
    temp[4] = 0
    temp[3] = index_four

    l.append(temp)

    temp = list(state)
    index_six = temp[6]
    temp[6] = 0
    temp[3] = index_six

    l.append(temp)

    l = sorted(l)
    return l
  if i == 4: # index 1, 3, 5 and 7 can be successors
    temp = list(state)
    index_one = temp[1]
    temp[1] = 0
    temp[4] = index_one

    l.append(temp)

    temp = list(state)
    index_three = temp[3]
    temp[3] = 0
    temp[4] = index_three

    l.append(temp)

    temp = list(state)
    index_seven = temp[7]
    temp[7] = 0
    temp[4] = index_seven

    l.append(temp)

    temp = list(state)
    index_five = temp[5]
    temp[5] = 0
    temp[4] = index_five

    l.append(temp)

    l = sorted(l)
    return l
  if i == 5: # index 2, 4 and 8 can be successors
    temp = list(state)
    index_two = temp[2]
    temp[2] = 0
    temp[5] = index_two

    l.append(temp)

    temp = list(state)
    index_four = temp[4]
    temp[4] = 0
    temp[5] = index_four

    l.append(temp)

    temp = list(state)
    index_eight = temp[8]
    temp[8] = 0
    temp[5] = index_eight

    l.append(temp)

    l = sorted(l)
    return l
  if i == 6: # index 3 and 7 can be successors
    temp = list(state)
    index_three = temp[3]
    temp[3] = 0
    temp[6] = index_three

    l.append(temp)

    temp = list(state)
    index_seven = temp[7]
    temp[7] = 0
    temp[6] = index_seven

    l.append(temp)

    l = sorted(l)
    return l
  if i == 7: # index 4, 6 and 8 can be successors
    temp = list(state)
    index_four = temp[4]
    temp[4] = 0
    temp[7] = index_four

    l.append(temp)

    temp = list(state)
    index_six = temp[6]
    temp[6] = 0
    temp[7] = index_six

    l.append(temp)

    temp = list(state)
    index_eight = temp[8]
    temp[8] = 0
    temp[7] = index_eight

    l.append(temp)

    l = sorted(l)
    return l
  if i == 8: # index 5 and 7 can be successors
    temp = list(state)
    index_five = temp[5]
    temp[5] = 0
    temp[8] = index_five

    l.append(temp)

    temp = list(state)
    index_seven = temp[7]
    temp[7] = 0
    temp[8] = index_seven

    l.append(temp)

    l = sorted(l)
    return l
