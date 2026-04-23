from computational_graph import CompGraph, ValueNode
from computational_graph.nodes.arithmetic import AddNode, MultiplyNode, SquareNode
from computational_graph.visualization import generate_tikz

# create all ValueNode objects
x1 = ValueNode("x1")
x2 = ValueNode("x2")
x3 = ValueNode("x3")
q1 = ValueNode("q1")
q2 = ValueNode("q2")
f = ValueNode("f")

# create all Operator Node objects
mult = MultiplyNode(x1, x2, q1)
add = AddNode([q1, x3], q2)
square = SquareNode(q2, f)

# build the graph by declaring inputs and outputs as 2 lists of ValueNode objects
cg = CompGraph([x1, x2, x3], [f])

# test the graph with some random inputs, output should be 100.0
cg.forward([2.0, 4.0, 2.0])
print(cg)

print(generate_tikz(cg))
