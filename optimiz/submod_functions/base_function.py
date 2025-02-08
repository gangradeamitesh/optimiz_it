

from optimiz.optimizer_factory import OptimizerFactory

class BaseFunction:

    def __init__(self , selected_indices = None , gains = None, optimizer_type="naive_greedy" ) -> None:
        self.optimizer = OptimizerFactory.get_optimizer(self.optimizer_type)

    def fit(self , subset_size):
        print(self.optimizer_type)
        return self.optimizer.select(subset_size , self)



