"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

import miniflow as mf

x, y, z = (mf.Input() for _ in range(3))

f = mf.Add(x, y, z)
f_mul = mf.Mul(x, y, z)

feed_dict = {x: 10, y: 5, z: 20}

sorted_nodes = mf.topological_sort(feed_dict)
output_add = mf.forward_pass(f, sorted_nodes)
output_mul = mf.forward_pass(f_mul, sorted_nodes)

# NOTE: because topological_sort set the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
print("{} + {} + {} = {}, mul: {}(according to miniflow)".format(feed_dict[
    x], feed_dict[y], feed_dict[z], output_add, output_mul))
