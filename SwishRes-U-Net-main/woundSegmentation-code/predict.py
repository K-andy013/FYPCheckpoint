import cv2
#from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from tensorflow.keras.models import load_model


from models.DUNet import build_model

from utils.learning.metrics import dice_coef, precision, recall
from utils.BilinearUpSampling import BilinearUpSampling2D
from utils.io.data import load_data, save_results, save_rgb_results, save_history, load_test_images, DataGen
from utils.io.dataVal import DataGenValidation

# settings
input_dim_x = 224
input_dim_y = 224
color_space = 'rgb'
data_path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/data/Fuseg/'
#data_path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/data/azh/'

#save_model_name = "model-DUnet-Fuseg-aug-100ep"
#save_model_path = 'SOTA-models/Fuseg/'
save_model_name = "model-DUnet-AZHData-aug-100ep"
save_model_path = 'SOTA-models/azh/'

weight_file_name = save_model_name+ '.hdf5'
#max_samples = 5

save_result_path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/results-SOTA/Fuseg/crossCorpora/' + save_model_name + '/model-pred-masks/'
#save_result_path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/results-SOTA/azh/crossCorpora/' + save_model_name + '/model-pred-masks/'

data_gen = DataGenValidation(data_path, split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space=color_space)
#data_gen = DataGen(data_path, split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space=color_space)

x_test, test_label_filenames_list = load_test_images(data_path)
# Slice both to match max_samples and the order in data_gen
#x_test = x_test[:max_samples]
test_label_filenames_list = data_gen.x_test_file_list  # Use filenames from data_gen for perfect alignment


## get  DUNet Model
model, model_name = build_model(shape=(input_dim_x, input_dim_y, 3))

"""#Fuseg weight file
model = load_model(save_model_path + weight_file_name
               , custom_objects={'recall':recall,
                                 'precision':precision,
                                 'dice_coef': dice_coef
                                 })
"""""
#AZH weight file
model.load_weights(save_model_path + weight_file_name)



for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True):
    prediction = model.predict(image_batch, verbose=1)
    save_results(prediction, 'rgb', save_result_path, test_label_filenames_list)
    break
