
import torch
import numpy as np

class EarlyStoppingOnStabilization:
    def __init__(self, patience):
        self.patience = patience
        self.min_delta = 0.1
        self.best_sparsity = float('inf')
        self.stabilized_epochs = 0

    def should_stop(self, current_sparsity):
        """
        Check if the training should be stopped.

        Parameters:
            current_sparsity (float): The current sparsity value.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if abs(current_sparsity - self.best_sparsity) < self.min_delta:
            self.stabilized_epochs += 1
            if self.stabilized_epochs >= self.patience:
                return True
        else:
            self.stabilized_epochs = 0

        self.best_sparsity = current_sparsity
        return False