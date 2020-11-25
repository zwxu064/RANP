import maxflow
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
  print(maxflow.__version__)

  ##
  g = maxflow.GraphFloat(2, 2)
  nodes = g.add_nodes(2)
  g.add_edge(nodes[0], nodes[1], 1, 2)
  g.add_tedge(nodes[0], 2, 5)
  g.add_tedge(nodes[1], 9, 4)
  flow = g.maxflow()
  print('maxflow: {}'.format(flow))

  for i in range(g.get_node_num()):
    print('seg of node {}: {}'.format(i, g.get_segment(i)))

  ##
  g = maxflow.GraphFloat()
  node_idxs = g.add_grid_nodes((1, 2))
  g.add_grid_edges(node_idxs, 50)
  g.add_grid_edges(node_idxs, 1, 3)
  g.maxflow()
  seg = g.get_grid_segments(node_idxs)
  print(seg)
  img = np.int_(np.logical_not(seg))
  plt.imshow(img)
  # plt.show(block=False)

  ##
  g = maxflow.GraphFloat()
  time_start = time.time()
  nodes = g.add_nodes(10000000)

  for i in range(g.get_node_num()):
    g.add_tedge(nodes[i], 1, 2)

  time_set = time.time() - time_start
  time_start = time.time()

  g.maxflow()
  time_solve = time.time() - time_start
  print('time, set: {:.4f}s, solve: {:.4f}s' \
        .format(time_set, time_solve))
  print(g.get_node_num())

  g.add_nodes(2000)
  print(g.get_node_num())
