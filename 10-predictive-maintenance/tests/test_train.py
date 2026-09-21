from src.train import run
def test_run():
    r=run()
    assert 0<=r["balanced_accuracy"]<=1
    assert 0<=r["f1"]<=1
