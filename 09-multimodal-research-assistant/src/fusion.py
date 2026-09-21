import numpy as np
from sklearn.preprocessing import StandardScaler
def text_features(texts):
    vocab=["ai","quantum","health","machine","research"]
    return np.array([[t.lower().split().count(w) for w in vocab] for t in texts],float)
def image_features(images):
    out=[]
    for img in images:
        a=np.asarray(img,dtype=float)/255
        out.append([a.mean(),a.std(),*a.mean(axis=(0,1))])
    return np.array(out)
def fuse(text_x,image_x):
    return np.c_[StandardScaler().fit_transform(text_x),StandardScaler().fit_transform(image_x)]
