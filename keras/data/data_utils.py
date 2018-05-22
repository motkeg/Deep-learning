import pandas as pd
import os
import shutil

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


TRAIN = ""
TEST  = ""

def Create_scale_data_files(trainPath,testPath , trunck=False):

    # check if fiels are not created yet
    train_p , train_n = os.path.split(trainPath)
    test_p ,test_n  =  os.path.split(testPath)
    if not trunck:
        if ( os.path.isfile(train_p+'/scale_'+train_n)  or os.path.isfile(test_p+'/scale_'+test_n)):
            print ("scaled files are Exists in {} or {} :".format( train_p,test_p))
            return False

    try:
        train_data = pd.read_csv(trainPath)
        test_data  = pd.read_csv(testPath)
    except FileNotFoundError as e:
        print(e)
        return False


    scaler  = MinMaxScaler(feature_range=(0,1))  ## scale the data to 0-1 values for best training

    scale_train_data = scaler.fit_transform(train_data)
    scale_test_data  = scaler.transform (test_data) # we use 'transform because it nedded to be sane as train_data'
    # Print out the adjustment that the scaler applied to the total_earnings column of data
    print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(
                                                                                                       scaler.scale_[8],
                                                                                                        scaler.min_[8]))
    # create new data frames for the scaled data and save it for use
    train_df  = pd.DataFrame(scale_train_data , columns=train_data.columns.values )
    test_df   = pd.DataFrame(scale_test_data , columns=test_data.columns.values)  

    TRAIN = train_p+'/scale_'+train_n
    TEST  = test_p+'/scale_'+test_n
    
    train_df.to_csv(train_p+'/scale_'+train_n)
    test_df.to_csv(test_p+'/scale_'+test_n)  

    print ("Scaled files are saved in : {} , {}".format('scale_'+train_n,
                                                        'scale_'+test_n  ) )  

    return train_df , test_df

def get_df():
    try:
        if (TRAIN != "" and TEST != ""):
            train_df = pd.read_csv(TRAIN)
            test_df  = pd.read_csv(TEST)
            return train_df , test_df
    except FileNotFoundError as e:
        print(e)
        print ("you need to do scaling before ,try :  Create_scale_data_files()")
        return False


def export_model_to_GCP(name , inputs , outputs , sess):
    
    if os.path.isdir('keras/export/'+name):
        shutil.rmtree('keras/export/'+name)

    model_builder = tf.saved_model.builder.SavedModelBuilder('keras/export/'+name)

    inputs = {"inputs": tf.saved_model.utils.build_tensor_info(inputs)}
    outputs = {'earnings': tf.saved_model.utils.build_tensor_info(outputs)}

    signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    model_builder.add_meta_graph_and_variables(
        sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature
        })

    model_builder.save()

# train, test = Create_scale_data_files('data/train.csv', 'data/test.csv',True)    
# X = train.drop('total_earnings' , axis=1)
# y = train[['total_earnings']]


# print(X.head())
# print(y.head())
# #get_df()


