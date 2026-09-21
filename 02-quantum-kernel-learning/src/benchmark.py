from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def run(seed=42):
    X,y=make_moons(n_samples=240,noise=.18,random_state=seed)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=seed)
    model=make_pipeline(StandardScaler(),SVC(kernel="rbf"))
    model.fit(Xtr,ytr)
    p=model.predict(Xte)
    return {"accuracy":accuracy_score(yte,p),"balanced_accuracy":balanced_accuracy_score(yte,p),"f1":f1_score(yte,p)}

if __name__=="__main__":
    print(run())
