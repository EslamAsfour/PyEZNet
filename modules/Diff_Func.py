# Generic Class for the classes that needs to calc Grads

class Diff_Func:
    """
        Abstract Class to represent the Construction of the Differientiable Functions
    """

    def __init__(self):
        # Init dic for the cached values and grads
        self.cache = {}
        self.grad = {}

    def __call__(self, *arg, **kargs):
        # Calc Forward path
        output = self.forward(*arg, **kargs)
        # Calc grads
        self.grad = self.calc_Grad(*arg, **kargs)

        return output

    def forward(self, *arg, **kargs):
        """
          Calc output
        """
        pass


    def calc_Grad(self, *arg, **kargs):
        """
           Calc Grads
        """
        pass


    def backward(self, *arg, **kargs):
        """
           Backward path
        """


        pass

