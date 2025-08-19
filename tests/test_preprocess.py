import pandas as pd
from amazon_logreg_pipeline import preprocess

def test_preprocess_drops_empty_and_binarizes():
    df = pd.DataFrame({
        "rating": [5, 2, None],
        "review_content": ["Good", "", "Bad"]
    })
    out = preprocess(df)
    assert "text" in out.columns
    assert out["sentiment"].tolist() == [1, 0]  # предполагается что пустая строк удалена
