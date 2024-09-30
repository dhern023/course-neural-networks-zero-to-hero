"""
Add a new function as an activation
and calculate back_propogation using a new class method
"""
import numpy
import pathlib

import micrograd

DIRNAME_OUT = "micrograd"
DIR_OUT = pathlib.Path(__file__).resolve().parents[1] / "out" / DIRNAME_OUT
DIR_OUT.mkdir(exist_ok=True, parents=True)

class Vertex(micrograd.Value):
    """
    Inherited class with added
        activation method
        back-propagation capability
    """
    def __init__(self, data, _children = (), _op="", label = ""):
        super().__init__(data, _children, _op, label)
        self._backward = lambda : None

    def __add__(self, instance_value):
        """ 
        overrides a + b for a,b Value instances 
        updated to calculate the gradient
        """
        out = self.__class__(self.data + instance_value.data, (self, instance_value), "+")

        def _backward():
            """
            dO/self
            = dO/dout * dout/dself
            = out.grad * d(self.data + instance_value.data)/dself
            = out.grad * 1
            """
            self.grad += out.grad * 1.0
            instance_value.grad += out.grad * 1.0

        out._backward = _backward

        return out

    def __mul__(self, instance_value):
        """
        overrides a * b for a,b Value instances
        updated to calculate the gradient
        """
        out = self.__class__(self.data * instance_value.data, (self, instance_value), "*")

        def _backward():
            """
            dO/self
            = dO/dout * dout/dself
            = out.grad * d(self.data * instance_value.data)/dself
            = out.grad * instance_value.data
            """
            self.grad += out.grad * instance_value.data
            instance_value.grad += out.grad * self.data

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = numpy.tanh(x)
        out = self.__class__(data=t, _children = (self,), label = "tanh")

        def _backward():
            """
            dO/self
            = dO/dout * dout/dself
            = out.grad * d tanh(x) / dself
            = out.grad * whatever that is
            """
            self.grad += out.grad * (1 - t**2)

        out._backward = _backward

        return out

    def backward(self):
        """
        Call the _backward instance at every step via BFS
        NOTE: This may also be considered topological order
        """
        self.grad = 1.0 # dR/dR = 1

        stack = [self]
        while stack:
            v = stack.pop()
            v._backward()
            for v_child in v._prev:
                stack.append(v_child)
        
        return self

def neuron(x, w, b):
    """
    NOTE: We set the label to 1-index to match the video
    """
    assert len(x) == len(w), f"Incompatible shapes: len(x) = {len(x)} and len(w) = {len(w)}"

    # Calculate inner product
    list_xiwi = []
    for i in range(len(x)):
        xi = Vertex(x[i], label = f"x{i+1}")
        wi = Vertex(w[i], label = f"w{i+1}")
        xiwi = xi*wi; xiwi.label = f"x{i+1}w{i+1}"
        list_xiwi.append(xiwi)

    L = list_xiwi[0]
    for i in list_xiwi[1:]:
        L += i
    L.label = "sum(xiwi)"

    bias = Vertex(b, label = "b")
    n = L + bias; n.label = "n"

    return n

x = [2.0, 0.0] # inputs
w = [-3.0, 1.0] # weights
b = 6.8813735870195432 # bias TODO: Figure out how they got it
n = neuron(x, w, b)
o = n.tanh(); o.label = "o"
micrograd.draw_dot(o).render(DIR_OUT / 'neuron-fp.gv', view=True)
micrograd.draw_dot(o.backward()).render(DIR_OUT / 'neuron-bp.gv', view=True)

# BUG: We used to overwrite the gradient instead of accumulating it.
a = Vertex(3.0, label = "a")
b = a + a; b.label = "b";
micrograd.draw_dot(b.backward()).render(DIR_OUT / "bug-bp.gv", view = True)

