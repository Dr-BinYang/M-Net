import numpy as np

class GlobalFeatureScaler:
    """Universal multi-dimensional global feature normalization processor

    Features:
    1. Supports data of any dimension (requires specifying feature channel axis)
    2. Automatically handles zero-variance features
    3. Unified normalization/inverse normalization interface
    """

    def __init__(self, epsilon=1e-8, feature_axis=-1):
        """
        Parameters:
        epsilon : Minimum standard deviation threshold to prevent division by zero errors
        feature_axis : Axis index where feature dimensions are located
        """
        self.epsilon = epsilon
        self.feature_axis = feature_axis
        self.means = None
        self.stds = None

    def fit(self, data):
        """Fit data to compute statistics

        Parameters:
        data : numpy.ndarray Input data, can have any shape but must contain feature axis
        """
        # Determine aggregation axes (excluding feature axis)
        other_axes = tuple(i for i in range(data.ndim) if i != self.feature_axis)

        # Compute statistics
        self.means = np.mean(data, axis=other_axes, keepdims=True)
        self.stds = np.std(data, axis=other_axes, keepdims=True)

        # Standard deviation protection
        self.stds = np.where(self.stds < self.epsilon, 1.0, self.stds)

    def transform(self, data):
        """Apply normalization"""
        return (data - self.means) / self.stds

    def inverse_transform(self, data):
        """Apply inverse normalization"""
        return data * self.stds + self.means