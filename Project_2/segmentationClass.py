import numpy as np
import PIL.Image as Image

class segmentationClass():
  def __init__(self):
    self.p0 = 1
    # user-set 1x2 array for foreground pixel
    self.x_a = np.array([3, 4]) # foreground pixel
    self.x_b = np.array([0, 0]) # background pixel

  # here, input is an NxNx3 array of RGB values
  def constructAdjacencyMatrix(self, I):
    # get the colors of the source and sink pixels
    source_color = I[self.x_a[0], self.x_a[1]]
    sink_color = I[self.x_b[0], self.x_b[1]]
    print(f"source color: {source_color}, sink color: {sink_color}")

    # create the adjacency matrix with all 0's
    rows, cols = I.shape[0], I.shape[1]
    num_pixels = int(rows * cols)
    A = np.zeros((num_pixels + 2, num_pixels + 2))
    # inner function to calculate the linear index in the adjacency matrix
    def index(row, column):
      return int(row * cols + column)

    for row in range(rows):
      for col in range(cols):
        # get x-value of index image[row, col] --> for the adjacency matrix
        current_index = index(row, col)
        if row > 0: # if not top row
          A[current_index, index(row - 1, col)] = self.p0  # Top pixel
          if col > 0: # if not leftmost column
            A[current_index, index(row - 1, col - 1)] = self.p0  # Top left pixel
          if col < cols - 1: # if not rightmost column
            A[current_index, index(row - 1, col + 1)] = self.p0  # Top right pixel
        if row < rows - 1: # if not bottom row
          A[current_index, index(row + 1, col)] = self.p0  # Bottom pixel
          if col > 0: # if not leftmost column
            A[current_index, index(row + 1, col - 1)] = self.p0  # Bottom left pixel
          if col < cols - 1: # if not rightmost column
            A[current_index, index(row + 1, col + 1)] = self.p0  # Bottom right pixel
        if col > 0: # if not leftmost column
          A[current_index, index(row, col - 1)] = self.p0  # Left pixel
        if col < cols - 1: # if not rightmost column
          A[current_index, index(row, col + 1)] = self.p0  # Right pixel
        
        # set the capacities to the source and sink nodes, based on RGB distance from current pixel
        sink = 442 - round(np.linalg.norm(I[row, col] - sink_color))
        source = 442 - round(np.linalg.norm(I[row, col] - source_color))
        A[current_index, -2] = sink   # Capacity to source
        A[current_index, -1] = source # Capacity to sink
        A[-1, current_index] = source # Capacity from source to pixel
        A[-2, current_index] = sink   # Capacity from sink to pixel
    return A
  
  def segmentImage(self, image):
    # output is NxNx1 array of 0s and 1s
    L = np.zeros((image.shape[0], image.shape[1]))
    # get pre-FF graph
    matrix = self.constructAdjacencyMatrix(image)
    # Ford it up, with a side of Fulkerson
    residual_graph = self.fordFulkerson(matrix, matrix.shape[0] - 2, matrix.shape[0] - 1)
    # set all reachable nodes to 1, else 0
    for row in range(image.shape[0]):
      for col in range(image.shape[1]):
        # get x-value of index image[row, col] --> for the adjacency matrix
        if self.BFS(residual_graph, matrix.shape[0] - 2, row*image.shape[1] + col, np.zeros(len(residual_graph), dtype=int)):
          L[row, col] = 1
    return L
    
  def fordFulkerson(self, graph, source, sink):
    # parent array to store the path
    parent = np.zeros(len(graph), dtype=int)
    residual_graph = np.copy(graph)  # Create a copy for the residual graph
    # while there is a path from source to sink
    while self.BFS(residual_graph, source, sink, parent):
      # set initial flow to infinity
      flow = np.inf
      # trace path from sink back to source
      s = sink
      while s != source:
        # find bottleneck capacity
        flow = np.minimum(flow, residual_graph[parent[s]][s])
        # go back one step towards source
        s = parent[s]

      # update residual capacities of the edges
      s = sink
      while s != source:
        t = parent[s]
        # subtract bottleneck capacity from residual capacity
        residual_graph[t][s] -= flow
        # add bottleneck capacity to forward capacity
        residual_graph[s][t] += flow
        s = parent[s]
    return residual_graph

  # short and efficient BFS implementation, TAKEN FROM GITHUB
  def BFS(self, graph, source, sink, parent):
    visited = [False] * len(graph)
    queue = []
    queue.append(source)
    visited[source] = True
    while queue:
      u = queue.pop(0)
      for index, val in enumerate(graph[u]):
        if visited[index] == False and val > 0:
          queue.append(index)
          visited[index] = True
          parent[index] = u
    return True if visited[sink] else False

# BTW, the multi-collinearity trap in the Kaggle competition was DIABOLICAL