import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def evaluate(seed=42, noise=.05):
    X,y=make_classification(n_samples=400,n_features=6,n_informative=4,random_state=seed)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=seed)
    m=LogisticRegression(max_iter=1000).fit(Xtr,ytr)
    clean=accuracy_score(yte,m.predict(Xte))
    rng=np.random.default_rng(seed)
    pert=Xte+rng.normal(0,noise,Xte.shape)
    robust=accuracy_score(yte,m.predict(pert))
    return {"clean_accuracy":clean,"perturbed_accuracy":robust,"accuracy_drop":clean-robust}

if __name__=="__main__":
    print(evaluate())
