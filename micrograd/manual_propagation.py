"""
Showing how propagation works forwards and backwards
using a simple expression

NOTE: All of the imports will be at the top, but the
functions will be defined in each "section" for educational purposes.
In practice, all functions should be at the top
"""
import pathlib

import micrograd

DIR_OUT = pathlib.Path(__file__).resolve().parents[1] / "out" / "micrograd"
DIR_OUT.mkdir(exist_ok=True, parents=True)

# Forward Propogation =============================================================================

def simple_example(x1, x2, x3, x4):
    """
    L = w6( w3 + (w1*w2) )
    """
    w1 = micrograd.Value(x1, label = "w1")
    w2 = micrograd.Value(x2, label = "w2")
    w3 = micrograd.Value(x3, label= "w3")
    w6 = micrograd.Value(x4, label = "w6")

    w4 = w1*w2; w4.label = "w4"
    w5 = w3 + w4; w5.label = "w5"
    L = w5*w6; L.label = "L"

    return L

x = [2.0, -3.0, 10.0, -2.0]
L = simple_example(*x)
micrograd.draw_dot(L).render(DIR_OUT / 'simple-example-fp.gv', view=True)

# Backward Propogation ============================================================================

def back_prop(root):
    """
    take partial(root)/partial(vertex) at each step

    Details:
        outcome = operation(inputs)
        To update input,
            other_inputs = inverse_operation(outcome, input)
            outcome_new = operation(input + h, other_inputs)
        Finally, 
            (outcome_new - outcome) / h

    Example:
        outcome = product( x1, x2, x3, ..., xn)
        Want to update x2?
            other_inputs = (outcome / x2) 
                         = product( x1, x3, ..., xn) without x2
            x2_new = x2+h
            outcome_new = product(x1, x3, ..., xn)*(x2_new)
        Finally,
            (outcome_new - outcome) / h

    WARNING: Overwrites the attributes of the original vertex
    """
    h = 0.001 # TODO: decrease the step size
    root.grad = 1 # dR/dR = 1
    stack = [root]
    while stack:
        v_current = stack.pop()
        if v_current._op: # then you have child vertices
            for v in v_current._prev: # calculate gradient of each child
                if v_current._op == "*":
                    v.grad = v_current.grad*((v_current.data / v.data) * (v.data + h) - v_current.data) / h
                elif v_current._op == "+":
                    v.grad = v_current.grad*((v_current.data - v.data) + (v.data + h) - v_current.data) / h
                stack.append(v)

    return root

micrograd.draw_dot(back_prop(L)).render(DIR_OUT / 'value-graph-bp.gv', view=True)

# What effect do the inputs have on the outcome? ==================================================

def affect_outcome(x, size_shift = 0.01):
    """
    Increase outcome by going into the gradient's direction
    Decrease outcome by going against the gradient's direction
    """
    L_original = back_prop(simple_example(*x)) # construct L's graph and compute gradients
    print("Original outcome: ", L_original.data)

    # Get all L's original vertices to update them
    vertices, edges = micrograd.trace(L_original)
    vertices2 = vertices.copy()

    # Increase L by going in the gradient's direction
    w1 = next(w for w in vertices if w.label == "w1")
    w2 = next(w for w in vertices if w.label == "w2")
    w3 = next(w for w in vertices if w.label == "w3")
    w6 = next(w for w in vertices if w.label == "w6")
    # Reassignment to prevent ovewriting
    w1 = micrograd.Value(w1.data + size_shift * w1.grad)
    w2 = micrograd.Value(w2.data + size_shift * w2.grad)
    w3 = micrograd.Value(w3.data + size_shift * w3.grad)
    w6 = micrograd.Value(w6.data + size_shift * w6.grad)

    w4 = w1*w2
    w5 = w3 + w4
    L = w5*w6
    print("Increased outcome: ", L.data)

    # Decrease L by going against the gradient's direction
    w1 = next(w for w in vertices2 if w.label == "w1")
    w2 = next(w for w in vertices2 if w.label == "w2")
    w3 = next(w for w in vertices2 if w.label == "w3")
    w6 = next(w for w in vertices2 if w.label == "w6")
    # Reassignment to prevent ovewriting
    w1 = micrograd.Value(w1.data - size_shift * w1.grad)
    w2 = micrograd.Value(w2.data - size_shift * w2.grad)
    w3 = micrograd.Value(w3.data - size_shift * w3.grad)
    w6 = micrograd.Value(w6.data - size_shift * w6.grad)

    w4 = w1*w2
    w5 = w3 + w4
    L = w5*w6
    print("Decreased outcome: ", L.data)

    return

size_shift = 0.01
affect_outcome(x,size_shift)