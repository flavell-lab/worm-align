import numpy as np
import random


GLOBAL_RANDOM_STATE = np.random.RandomState(47)

class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be
    either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent
    between raw and labeled datasets, otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across
    (1,2) axis)
    """

    def __init__(self, random_state, axes=[(1,2)], force_180=False, **kwargs):

        self.random_state = random_state
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0
        self.axes = axes
        self.force_180 = force_180

    def __call__(self, image):

        axis = self.axes[self.random_state.randint(len(self.axes))]
        assert image.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        k = self.random_state.choice([0, 2])
        # rotate k times around a given plane
        if image.ndim == 3:
            image = np.rot90(image, k, axis)
        else:
            channels = [np.rot90(image[c], k, axis) for c in range(image.shape[0])]
            image = np.stack(channels, axis=0)

        return image


