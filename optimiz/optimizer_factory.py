from .gradient_descent import GradientDescent
from .coordinate_descent import CoordinateDescent

class OptimizerFactory:
    """
    Factory class to create optimizer instances.

    Methods:
        get_optimizer(optimizer_type, **kwargs): Returns an optimizer instance based on the type.
    """

    @staticmethod
    def get_optimizer(optimizer_type, **kwargs):
        """
        Returns an optimizer instance based on the specified type.

        Args:
            optimizer_type (str): The type of optimizer to create.
            **kwargs: Additional arguments for the optimizer.

        Returns:
            Optimizier: An instance of the specified optimizer.
        """
        if optimizer_type == "gradient_descent":
            return GradientDescent(**kwargs)
        if optimizer_type == "coordinate_descent":
            return CoordinateDescent(**kwargs)