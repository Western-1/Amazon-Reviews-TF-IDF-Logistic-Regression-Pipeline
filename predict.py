from pathlib import Path
import joblib
import sys

def predict_one(text: str, model_path="artifacts/logreg_model.pkl", vec_path="artifacts/tfidf_vectorizer.pkl"):
    vec = joblib.load(vec_path)
    model = joblib.load(model_path)
    X = vec.transform([text])
    proba = model.predict_proba(X)[0,1]
    return {"proba": float(proba), "pred": int(proba>=0.5)}

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "sample review text"
    print(predict_one(text))
