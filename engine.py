import math
import random


class Value:
    def __init__(self, data, parents=None, operation='', label=''):
        self.data = data
        self.grad = 0.0
        self.parents = parents if parents else ()
        self.operation = operation
        self.label = label

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data + other.data, parents=(self, other), operation='+')

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data * other.data, parents=(self, other), operation='*')

    def tanh(self):
        return Value(math.tanh(self.data), parents=(self,), operation='tanh')
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other
    
    def square(self):
        return self * self

    def backward(self):
        topo = []
        visited = set()
        self.grad = 1.0

        def build(value):
            if value in visited: return
            
            visited.add(value) 
            for parent in value.parents:
                build(parent)
            topo.append(value)

        build(self)
        topo.reverse()
        for value in topo:
            value.backward_pass()

    def backward_pass(self):
        if self.parents:
            if self.operation == '+':
                for parent in self.parents:
                    parent.grad += self.grad

            elif self.operation == '*':
                self.parents[0].grad += self.parents[1].data * self.grad
                self.parents[1].grad += self.parents[0].data * self.grad
            
            elif self.operation == 'tanh':
                self.parents[0].grad += (1 - self.data**2) * self.grad


class Neuron:
    def __init__(self, n_inputs):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        result = 0.0
        for weights, inputs in zip(self.weights, x):
            result += weights * inputs
        return (result + self.bias).tanh()
    
    def __repr__(self):
        return f'Weights: {self.weights}, Bias: {self.bias}'
    
    def parameters(self):
        return self.weights + [self.bias]
    

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        result = []
        for neuron in self.neurons:
            result.extend(neuron.parameters())
        return result
    

class MLP:
    def __init__(self, sizes):
        self.layers = [
            Layer(n_inputs, n_outputs)
            for n_inputs, n_outputs in zip(sizes, sizes[1:])
            ]
    
    def __call__(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output
    
    def parameters(self):
        result = []
        for layer in self.layers:
            result.extend(layer.parameters())
        return result

    


random.seed(42)
xs = [0.5]
ys = [0.2]

mlp = MLP(sizes=[1, 1, 1])
lr = 0.1
e = 1e-6
losses = []

param = mlp.parameters()[0]

param.data = param.data + e
result = mlp(xs)
loss = sum((res - expected).square() for res, expected in zip(result, ys))
losses.append(loss.data)

param.data = param.data - (e * 2)
result = mlp(xs)
loss = sum((res - expected).square() for res, expected in zip(result, ys))
losses.append(loss.data)

param.data = param.data + e
result = mlp(xs)
loss = sum((res - expected).square() for res, expected in zip(result, ys))
loss.backward()



numeric_grad = (losses[0] - losses[1]) / (2*e)

print(numeric_grad, param.grad)


# for step in range(1):
#     result = []
#     for x in xs:
#         result.extend(mlp(x))

#     loss = sum((res - expected).square() for res, expected in zip(result, ys))

#     loss.backward()
    
#     params = mlp.parameters()
#     for param in params:
#         param.data = param.data - param.grad * lr
#         param.grad = 0.0
    
#     if step % 50 == 0:
#         print(f'Step: {step}, loss: {loss.data}')

#     if step % 250 == 0:
#         for res, expected in zip(result, ys):
#             print(f'predicted: {res} | expected: {expected}')

    

