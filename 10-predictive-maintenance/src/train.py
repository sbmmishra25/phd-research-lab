import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,balanced_accuracy_score

def make_data(seed=42,machines=6,samples_per_machine=120):
    rng=np.random.default_rng(seed); rows=[]; labels=[]; groups=[]
    for machine in range(machines):
        X=rng.normal(size=(samples_per_machine,4))+rng.normal(0,.35,4)
        risk=X[:,0]+.7*X[:,1]-.5*X[:,2]
        y=(risk>np.quantile(risk,.72)).astype(int)
        rows.append(X); labels.append(y); groups += [machine]*samples_per_machine
    return np.vstack(rows),np.concatenate(labels),np.array(groups)

def run(seed=42):
    X,y,g=make_data(seed)
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=2,random_state=seed).split(X,y,g))
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(X[tr],y[tr])
    p=m.predict(X[te])
    return {"balanced_accuracy":balanced_accuracy_score(y[te],p),"f1":f1_score(y[te],p,zero_division=0)}
if __name__=="__main__": print(run())
