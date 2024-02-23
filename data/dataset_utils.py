from data.data_interface import DataInterface
import pandas as pd
from aif360.sklearn.datasets import fetch_adult

def get_dataset_interface_by_name(name: str) -> DataInterface:
    if name == "credit default":
        return create_credit_default_interface(name)
    elif name == "give credit":
        return create_give_credit_interface(name)
    elif name == "adult census":
        return create_census_interface(name)
    else:
        raise Exception("Invalid dataset name.")


def create_credit_default_interface(name) -> DataInterface:
    continuous_features = ["LIMIT_BAL", "AGE"]+[f"PAY_{i}" for i in range(
        1, 7)]+[f"BILL_AMT{i}" for i in range(1, 7)]+[f"PAY_AMT{i}" for i in range(1, 7)]
    ordinal_features = ["EDUCATION"]
    ordinal_features_order = {"EDUCATION": [5,4,3, 2, 1]}
    unidirection_features = [[], ["EDUCATION"]]
    categorical_features = ["SEX", "MARRIAGE"]
    immutable_features = ["SEX", "MARRIAGE","AGE"]
    label_column = "default payment next month"
    positive_label = 0
    file_path = (
        "data/datasets/default of credit card clients.xls")
    df = pd.read_excel(file_path, header=1)
    df.loc[df['EDUCATION'].isin([0,6]), "EDUCATION"] = 5

    df.rename(columns={"PAY_0": "PAY_1"}, inplace=True)
    dropped_columns=["ID"]

    for pay_column in [f"PAY_{i}" for i in range(1, 7)]:
        df.loc[df[pay_column] < 0, pay_column] = 0
    di = DataInterface(df, None, continuous_features, ordinal_features,
                           categorical_features, immutable_features, label_column,
                           pos_label=positive_label, dropped_columns=dropped_columns, 
                           unidirection_features=unidirection_features, 
                           ordinal_features_order=ordinal_features_order,data_name=name)
    return di

def create_give_credit_interface(name) -> DataInterface:
    continuous_features = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ]
    ordinal_features = []
    ordinal_features_order = {}
    unidirection_features = [[], []]
    categorical_features = []
    immutable_features = ["age"]
    dropped_columns=["ID"]
    label_column = "SeriousDlqin2yrs"
    positive_label = 0
    file_path = (
        "data/datasets/give_credit.csv")
    df = pd.read_csv(file_path)
    df = df.dropna()
    di = DataInterface(df, None, continuous_features, ordinal_features,
                           categorical_features, immutable_features, label_column,
                           pos_label=positive_label, dropped_columns=dropped_columns, 
                           unidirection_features=unidirection_features, ordinal_features_order=ordinal_features_order
                           ,data_name=name)
    return di

def create_census_interface(name):
    """
    Loads and preprocesses the Adult Census Income dataset using the AIF360 library.
    AIF360 data functions returns Pandas dataframes with the protected
    attribute(s) encoded in the index.

    """
    X, y, _ = fetch_adult(subset="all")
    df = pd.concat([X, y], axis=1).reset_index(drop=True)
    df['annual-income'] = y.factorize(sort=True)[0]
    df = df.dropna()

    continuous_features = [
        'age',
        'education-num',
        'capital-gain',
        'capital-loss', 
        'hours-per-week'
    ]
    ordinal_features = []
    ordinal_features_order = {}
    unidirection_features = [[], ["education-num"]]
    categorical_features = ['occupation','workclass','marital-status', 'relationship', 'race', 'sex', 'native-country']
    immutable_features =  ['age','marital-status', 'relationship', 'race', 'sex', 'native-country'] 
    label_column = "annual-income"
    positive_label = 1
    dropped_columns = ['education']
    di = DataInterface(df, None, continuous_features, ordinal_features,
                            categorical_features, immutable_features, label_column,
                            pos_label=positive_label, dropped_columns=dropped_columns, 
                            unidirection_features=unidirection_features, 
                            ordinal_features_order=ordinal_features_order,data_name=name)
    return di