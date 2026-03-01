import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/raw/creditcard.csv")
    p.add_argument("--out_dir", type=str, default="data/prepared")

    p.add_argument("--target", type=str, default="Class")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)

    # feature engineering (optional)
    p.add_argument("--add_hour", action="store_true",
                   help="Add Hour feature computed from Time (Time%86400)/3600")

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in columns")

    # Optional: Hour feature from Time
    if args.add_hour and "Time" in df.columns:
        df["Hour"] = ((df["Time"] % 86400) / 3600).astype(float)

    # Stratified split to preserve fraud ratio
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.target].astype(int),
    )

    train_path = os.path.join(args.out_dir, "train.csv")
    test_path = os.path.join(args.out_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Prepared data saved:")
    print(f"- {train_path}  shape={train_df.shape}")
    print(f"- {test_path}   shape={test_df.shape}")


if __name__ == "__main__":
    main()