import  pandas as pd

# Load training data set from CSV file
def get_data():
    df = pd.read_csv("creditcard.csv", dtype=float)
    df = df.sample(frac=1)
    train_data = df[:200000]
    test_data  = df[200001:250000]
    eavl_data  = df[250001:]


    # Pull out columns for X (data to train with) and Y (value to predict)
    X_training = train_data.drop('Class', axis=1).values
    Y_training = train_data[['Class']].values

    # Pull out columns for X (data to test with) and Y (value to predict)
    X_testing = test_data.drop('Class', axis=1).values
    Y_testing = test_data[['Class']].values

    # Pull out columns for X (data to evaluate with) and Y (value to predict)
    X_eval = eavl_data.drop('Class', axis=1).values
    Y_eval = eavl_data[['Class']].values
    return {"X":X_training,"Y":Y_training} , {"X":X_testing,"Y":Y_testing} ,{"X":X_eval,"Y":Y_eval}