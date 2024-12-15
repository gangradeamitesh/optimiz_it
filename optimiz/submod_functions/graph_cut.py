from .base_function import BaseFunction


class GraphCut(BaseFunction):

    def __init__(self,X=None, selected_indices=None, gains=None, optimizer_type="naive_greedy") -> None:
        super().__init__(selected_indices, gains, optimizer_type)
        self.X = X
        