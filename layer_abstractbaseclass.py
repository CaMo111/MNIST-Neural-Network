
class Layer:
    '''
    Whenever we get input, we get an output. This is called
    forward propogation.
    There also exists Backward propogation, where we are updating the
    parameters. During backward propogation, we are given the deritive of the error in terms of the output
    but instead we are getting the derivitive of the erorr of the input.

    However since one layers output is equivilant to another layers input, 
    the derivitive of these errors will also be equal on both sides (see readme.md)
    '''
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def foward(self, input):
        # TODO: Return output
        pass

    def backward(self, output_gradient, learning_rate):
        #TODO update parameters (to make model better) and return input gradient (for next backward propogation iteration)
        pass
