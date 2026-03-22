import pandas as pd


def test_data_exists():
    df = pd.read_csv("data/prepared/train.csv")
    assert len(df) > 0


def test_target_exists():
    df = pd.read_csv("data/prepared/train.csv")
    assert "Class" in df.columns


def test_no_nulls():
    df = pd.read_csv("data/prepared/train.csv")
    assert df.isnull().sum().sum() == 0