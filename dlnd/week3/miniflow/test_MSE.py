"""
Test your MSE method with this script!

No changes necessary, but feel free to play
with this script to test your network.
"""

import numpy as np
import miniflow as mf

y, a = (mf.Input() for _ in range(2))
cost = mf.MSE(y, a)

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}
graph = mf.topological_sort(feed_dict)
# forward pass
output = mf.forward_pass(cost, graph)

"""
Expected output

23.4166666667
"""
print(output)
