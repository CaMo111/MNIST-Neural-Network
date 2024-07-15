from neuralnetwork import NeuralNetwork
import denselayer
import activationandlosses

network = NeuralNetwork()

network.layers = [
    denselayer.DenseLayer(28 * 28, 10),
    activationandlosses.TanH(),
    denselayer.DenseLayer(28 * 28, 10),
    activationandlosses.TanH()
]