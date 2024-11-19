

from optimiz.optimizer_factory import OptimizerFactory

class BaseFunction:

    def __init__(self , simi_matrix = None , selected_indices = None , gains = None , current_values = None , optimizer_type="naive_greedy" ) -> None:
        self.simi_matrix = simi_matrix
        self.selected_indices = selected_indices
        self.gain = None
        self.current_values = current_values
        self.optimizer_type = optimizer_type
        print(self.optimizer_type)
        self.optimizer = OptimizerFactory.get_optimizer(self.optimizer_type)

    def fit(self , subset_size):
        print(self.optimizer_type)
        self.optimizer.select(subset_size)



