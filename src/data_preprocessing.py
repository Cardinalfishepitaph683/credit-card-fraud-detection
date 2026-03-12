import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load fraud detection dataset."""
    df = pd.read_csv(file_path)
    return df


def main() -> None:
    file_path = "data/creditcard.csv"
    df = load_data(file_path)

    print("Dataset loaded successfully!")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nDataset info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nFraud percentage:")
    fraud_percentage = (df["Class"].sum() / len(df)) * 100
    print(f"{fraud_percentage:.4f}%")

    print("\nBasic statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()