import numpy as np
class LoRA:
    def __init__(self,in_features,out_features,rank=2,seed=42):
        rng=np.random.default_rng(seed)
        self.A=rng.normal(0,.02,(rank,in_features))
        self.B=np.zeros((out_features,rank))
    def delta(self): return self.B @ self.A
    @property
    def trainable_parameters(self): return self.A.size+self.B.size
if __name__=="__main__": print(LoRA(768,768,8).trainable_parameters)
