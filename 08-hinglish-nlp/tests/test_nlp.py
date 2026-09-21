from src.train import normalize
def test_normalize():
    assert normalize("  Hello   WORLD ")=="hello world"
