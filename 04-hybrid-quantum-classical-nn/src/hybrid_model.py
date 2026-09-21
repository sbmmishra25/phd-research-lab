import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def hybrid_features(X,theta):
    z=X*theta[:X.shape[1]]
    return np.c_[np.sin(z).sum(1),np.cos(z).sum(1),X]

def run(seed=42):
    X,y=make_moons(300,noise=.15,random_state=seed)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=seed)
    a,b=hybrid_features(Xtr,np.array([1,.8])),hybrid_features(Xte,np.array([1,.8]))
    m=LogisticRegression(max_iter=1000).fit(a,ytr)
    return accuracy_score(yte,m.predict(b))

if __name__=="__main__":
    print({"test_accuracy":run()})
