import pandas as pd


def test_data_exists():
    df = pd.read_csv("data/ci/train_sample.csv")
    assert len(df) > 0


def test_target_exists():
    df = pd.read_csv("data/ci/train_sample.csv")
    assert "Class" in df.columns


def test_no_nulls():
    df = pd.read_csv("data/ci/train_sample.csv")
    assert df.isnull().sum().sum() == 0