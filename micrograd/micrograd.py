import graphviz

class Value:
    """
    Wrapper for numerical objects that overrides the built-in operations
    Includes the children in order to build a tree structure that
        provides input and operation history
    """

    def __init__(self, data, _children = (), _op="", label = ""):
        """
        NOTE: Uses tuples since we shouldn't have a mutable parameter
        """
        self.data = data
        self.grad = 0.0 # assumes every value initially has no affect on output
        self._prev = set(_children) # storage and lookup efficiency
        self._op = _op
        self.label = label
 
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, instance_value):
        """ overrides a + b for a,b Value instances """
        out = self.__class__(self.data + instance_value.data, (self, instance_value), "+")
        return out
    
    def __mul__(self, instance_value):
        """ overrides a * b for a,b Value instances """
        out = self.__class__(self.data * instance_value.data, (self, instance_value), "*")
        return out

def trace(root):
    """ 
    Builds a set of a vertices and edges in the graph
    Returns tuple (vertices, edges) = (vertices, ((vertex, vertex)))

    Basically BFS with no destination in mind. 
    """
    stack = [root]
    vertices = []
    edges = []

    while stack:
        v = stack.pop()
        if v not in vertices:
            vertices.append(v)
            for v_child in v._prev:
                edges.append((v_child, v))
                stack.append(v_child)

    return vertices, edges

def draw_dot(root):
    """
    Create a rectangle for any vertex's value with a label
    Create a circle for the operation
    """
    dot = graphviz.Digraph(format = "svg", graph_attr={"rankdir": "LR"})

    vertices, edges = trace(root)
    # vertices = set(vertices)
    # edges = set(edges)
    
    for v in vertices:
        uid = str(id(v))
        dot.node(name = uid, label = f"{{ {v.label} | data {v.data:4f} | grad {v.grad:4f} }}", shape = "record")
        if v._op:
            name = f"{uid}{v._op}"
            dot.node(name = name, label = v._op)
            dot.edge(tail_name=name, head_name = uid)
    
    for (v1, v2) in edges:
        dot.edge(tail_name = str(id(v1)), head_name = str(id(v2)) + v2._op)

    return dot
