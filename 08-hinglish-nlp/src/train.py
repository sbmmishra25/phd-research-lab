from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
ROOT=Path(__file__).resolve().parents[1]
def normalize(text): return " ".join(str(text).strip().lower().split())
def run():
    df=pd.read_csv(ROOT/"data"/"sample.csv")
    x=df.text.map(normalize); y=df.label
    v=TfidfVectorizer(ngram_range=(1,2)); X=v.fit_transform(x)
    m=LogisticRegression(max_iter=1000).fit(X,y)
    return classification_report(y,m.predict(X),zero_division=0)
if __name__=="__main__": print(run())
