from .base import BaseOverSampler
from sklearn.base import clone

class MetaOversampler(BaseOverSampler):
    def __init__(self, base_ovs, fallback_ovs):
        self.base_ovs = base_ovs
        self.ovs = clone(base_ovs)
        self.fallback_ovs = clone(fallback_ovs)

    def fit_resample(self, X, y):
        try:
            return self.ovs.fit_resample(X, y)
        except:
            return self.fallback_ovs.fit_resample(X, y)

    def _fit_resample(self, X, y):
        pass
