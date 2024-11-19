class SubModOptimizer:

    def __init__(self , function=None , gains=None) -> None:
        self.function = function
        self.gains = gains

    def select(self , subset_size):
        raise NotImplementedError
    
