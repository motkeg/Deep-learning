import numpy as np
from tensorflow.python import keras

from data import data_utils
from tensorflow.python.keras.applications import ResNet50 
from tensorflow.python.keras.applications.resnet50 import decode_predictions ,preprocess_input
from tensorflow.python.keras.preprocessing import image



model = ResNet50()


# you can use any jpg image you want, you need to resize it to 224x224.
img = image.load_img('data/bay.jpg' , target_size=(224,224))

x = image.img_to_array(img)
# addind extra dimention to feed the net
x = np.expand_dims(x , axis=0)

# scale x
#x = preprocess_input(x)

# export model for use in GCP

data_utils.export_model_to_GCP('resnet50_ext' , 
                                model.input ,
                                model.output,  
                                keras.backend.get_session())

predictions = model.predict(x)
decoded_pred = decode_predictions(predictions)

for pred in decoded_pred:
    print("\n-----the predictions are: -------\n" , *pred , sep="\n")
