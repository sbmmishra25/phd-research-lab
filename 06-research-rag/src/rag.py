from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ResearchRAG:
    def __init__(self,documents):
        self.documents=documents
        self.vectorizer=TfidfVectorizer(stop_words="english")
        self.matrix=self.vectorizer.fit_transform(documents)
    def retrieve(self,query,k=3):
        q=self.vectorizer.transform([query])
        scores=cosine_similarity(q,self.matrix).ravel()
        ids=scores.argsort()[::-1][:k]
        return [{"id":int(i),"score":float(scores[i]),"text":self.documents[i]} for i in ids]

if __name__=="__main__":
    docs=["Quantum kernels compare encoded quantum states.","RAG grounds generation in retrieved evidence.","Predictive maintenance uses sensor data."]
    print(ResearchRAG(docs).retrieve("quantum kernel"))
