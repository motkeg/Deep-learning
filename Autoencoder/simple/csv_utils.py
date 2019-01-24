import pandas as pd
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


RANDOM_SEED = 1945

class CSV():
    def __init__(self ,path):
        try:
            self.df = pd.read_csv(path, index_col=0).dropna(0)

        except FileNotFoundError  as e:
            print (e)   

    def to_train_test(self , test_size = 0.2):
        X_train, X_test = train_test_split(self.df, test_size=test_size, random_state = RANDOM_SEED)
        print(f"Train: { X_train.shape} | Test: {X_test.shape}" )
        return X_train, X_test

    def to_train_test_y(self , class_label,test_s = 0.2  ):
        X_train, X_test = train_test_split(self.df, test_size=test_s, random_state = RANDOM_SEED)
        try:
            Y_train = X_train[class_label]
            X_train = X_train.drop([class_label])
            Y_test =  X_test[class_label]
            X_train = X_test.drop([class_label])
            print(f"Train: { X_train.shape} | Test: {X_test.shape}" )
            return X_train,Y_train, X_test,Y_test

        except ValueError as e:
            print(e)    


    def drop(self,list_arg):
        return self.df.drop(list_arg )

    def normalaize(self) :
        scalar = MinMaxScaler()
        df_scaled = scalar.fit_transform(self.df)
        self.df  = df_scaled
       

           



