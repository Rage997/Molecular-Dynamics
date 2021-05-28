import numpy as np

class VerletList:
    def __init__(self, data):
        d = data.shape[0]
        self.list = np.array((d, d), ndmin=2, dtype=np.object_)