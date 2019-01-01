from matrix import *

mutation_rate = 0.2

class Net:

    def __init__(self, inputs, hidden, outputs):
        self.input_nodes = inputs
        self.hidden_nodes = hidden
        self.output_nodes = outputs

        self.ih_weights = Matrix(hidden,inputs+1)
        self.hh_weights = Matrix(hidden, hidden+1)
        self.ho_weights = Matrix(outputs,hidden+1)

        self.ih_weights.randomize()
        self.hh_weights.randomize()
        self.ho_weights.randomize()

    def compute(self, inputs):
        inp = Matrix.columnMatrixFromArray(inputs)
        inp.addBias()

        # Calculate the first layer
        layer_1 = self.ih_weights.multiplyByMatrix(inp)
        layer_1.activate()
        layer_1.addBias()

        # Calculate the second layer
        layer_2 = self.hh_weights.multiplyByMatrix(layer_1)
        layer_2.activate()
        layer_2.addBias()

        # Calculate the outputs
        outputs = self.ho_weights.multiplyByMatrix(layer_2)
        outputs.activate()

        return outputs.toArray()

    def mutate(self):
        self.ih_weights.mutate(mutation_rate)
        self.hh_weights.mutate(mutation_rate)
        self.ho_weights.mutate(mutation_rate)

    def clone(self):
        c = Net(self.input_nodes, self.hidden_nodes, self.output_nodes)
        c.ih_weights = self.ih_weights.clone()
        c.hh_weights = self.hh_weights.clone()
        c.ho_weights = self.ho_weights.clone()
        return c

e = Net(2,5,3)
print(e.compute([6,5]))