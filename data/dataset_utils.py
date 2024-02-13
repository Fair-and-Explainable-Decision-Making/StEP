from data.data_interface import DataInterface
import pandas as pd
from aif360.sklearn.datasets import fetch_adult

def get_dataset_interface_by_name(name: str) -> DataInterface:
    if name == "credit default":
        return create_credit_default_interface()


def create_credit_default_interface() -> DataInterface:
    continuous_features = ["LIMIT_BAL", "AGE"]+[f"PAY_{i}" for i in range(
        1, 7)]+[f"BILL_AMT{i}" for i in range(1, 7)]+[f"PAY_AMT{i}" for i in range(1, 7)]
    ordinal_features = []
    categorical_features = []
    immutable_features = []
    label_column = "default payment next month"
    positive_label = 0
    file_path = (
        "data/datasets/default of credit card clients.xls")
    df = pd.read_excel(file_path, header=1)
    df.rename(columns={"PAY_0": "PAY_1"},inplace=True)

    for pay_column in [f"PAY_{i}" for i in range(1, 7)]:
        df.loc[df[pay_column] < 0, pay_column] = 0
    di = DataInterface(None, file_path, continuous_features, ordinal_features,
                       categorical_features, immutable_features, label_column,
                       pos_label=positive_label, file_header_row=1, dropped_columns=["ID"])
    return di

def load_census():
    """
    Loads and preprocesses the Adult Census Income dataset using the AIF360 library.
    AIF360 data functions returns Pandas dataframes with the protected
    attribute(s) encoded in the index.

    Args:
        prot_attr: name of protected attribute
        train_size: percentage of data to be used for training
    """
    X, y, _ = fetch_adult(subset="all")
    df = pd.concat([X, y], axis=1).reset_index(drop=True)
    df['annual-income'] = y.factorize(sort=True)[0]
    return df.dropna()
print(load_census())
