import os
import cv2
import json
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from augmentation import Augmentation  # Adjust import path as needed


class DataGen:
   
   def shuffle_image_label_lists_together(self):
        combined = list(zip(self.image_file_list, self.label_file_list))
        random.shuffle(combined)
        return zip(*combined)
   
   def get_num_data_points(self, train=False, val=False):
        try:
            image_file_list = self.x_train_file_list if val is False and train is True else self.x_val_file_list
        except ValueError:
            print('one of train or val need to be True')

        return len(image_file_list)
   
   def generate_data(self, batch_size, train=False, val=False, test=False):
        """Replaces Keras' native ImageDataGenerator."""
        try:
            if train is True:
                image_file_list = self.x_train_file_list
                label_file_list = self.y_train_file_list
            elif val is True:
                image_file_list = self.x_val_file_list
                label_file_list = self.y_val_file_list
            elif test is True:
                image_file_list = self.x_test_file_list
                label_file_list = self.y_test_file_list
        except ValueError:
            print('one of train or val or test need to be True')

        i = 0
        while True:
            image_batch = []
            label_batch = []
            for b in range(batch_size):
                if i == len(self.x_train_file_list):
                    i = 0
                if i < len(image_file_list):
                    sample_image_filename = image_file_list[i]
                    sample_label_filename = label_file_list[i]
                    if train or val:
                        image = cv2.imread(self.path_train_images + sample_image_filename, 1)
                        label = cv2.imread(self.path_train_labels + sample_label_filename, 0)
                    elif test is True:
                        image = cv2.imread(self.path_test_images + sample_image_filename, 1)
                        label = cv2.imread(self.path_test_labels + sample_label_filename, 0)
                    label = np.expand_dims(label, axis=2)

                    # --- AUGMENTATION & NORMALIZATION ---
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    label_2d = label.squeeze()
                    if train:
                        aug_img, aug_mask = self.augmentor.apply_train(image, label_2d)
                        image = aug_img.permute(1, 2, 0).cpu().numpy()
                        label = aug_mask.cpu().numpy()
                        if label.ndim == 2:
                            label = np.expand_dims(label, axis=2)
                    else:  # val or test
                        image = self.augmentor.apply_test(image)
                        image = image.permute(1, 2, 0).cpu().numpy()
                        # For mask, just ensure correct shape and type
                        label = label_2d.astype("float32")
                        if label.ndim == 2:
                            label = np.expand_dims(label, axis=2)

                    image = image.astype("float32")
                    label = label.astype("float32")

                    if image.shape[0] == self.x and image.shape[1] == self.y:
                        image_batch.append(image)
                    else:
                        print('the input image shape is not {}x{}'.format(self.x, self.y))
                    if label.shape[0] == self.x and label.shape[1] == self.y:
                        label_batch.append(label)
                    else:
                        print('the input label shape is not {}x{}'.format(self.x, self.y))
                i += 1
            if image_batch and label_batch:
                yield (np.array(image_batch), np.array(label_batch))

   def __init__(self, path, split_ratio, x, y, color_space='rgb', max_samples=None):
    self.x = x
    self.y = y
    self.path = path
    print(self.path)
    self.color_space = color_space
    self.path_train_images = path + "train/images/"
    self.path_train_labels = path + "train/labels/"
    self.path_test_images = path + "test/images/"
    self.path_test_labels = path + "test/labels/"
    self.image_file_list = get_png_filename_list(self.path_train_images)
    self.label_file_list = get_png_filename_list(self.path_train_labels)
    self.image_file_list[:], self.label_file_list[:] = self.shuffle_image_label_lists_together()

    # Limit dataset size if max_samples is set
    if max_samples is not None:
        self.image_file_list = self.image_file_list[:max_samples]
        self.label_file_list = self.label_file_list[:max_samples]

    self.split_index = int(split_ratio * len(self.image_file_list))
    self.x_train_file_list = self.image_file_list[self.split_index:]
    self.y_train_file_list = self.label_file_list[self.split_index:]
    self.x_val_file_list = self.image_file_list[:self.split_index]
    self.y_val_file_list = self.label_file_list[:self.split_index]
    self.x_test_file_list = get_png_filename_list(self.path_test_images)
    self.y_test_file_list = get_png_filename_list(self.path_test_labels)
    self.augmentor = Augmentation()  # Add this line
    

    

    

    @staticmethod
    def change_color_space(image, label, color_space):
        if color_space.lower() in ['hsi', 'hsv']:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)
        elif color_space.lower() == 'lab':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2LAB)
        return image, label
def normalize(arr):
    diff = np.amax(arr) - np.amin(arr)
    diff = 255 if diff == 0 else diff
    arr = arr / np.absolute(diff)
    return arr


def get_png_filename_list(path):
    file_list = []
    for dirName, _, filenames in os.walk(path):
        for filename in filenames:
            # check file extension
            if filename.lower().endswith(".png"):
                file_list.append(filename)
    file_list.sort()
    return file_list


def get_jpg_filename_list(path):
    file_list = []
    for dirName, _, filenames in os.walk(path):
        for filename in filenames:
            # check file extension
            if filename.lower().endswith(".jpg"):
                file_list.append(filename)
    file_list.sort()
    return file_list




def load_jpg_images(path):
    file_list = get_jpg_filename_list(path)
    temp_list = []
    for filename in file_list:
        img = cv2.imread(path + filename, 1)
        temp_list.append(img.astype("float32"))

    temp_list = np.array(temp_list)
    # x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    return temp_list, file_list


def load_png_images(path):

    temp_list = []
    file_list = get_png_filename_list(path)
    for filename in file_list:
        img = cv2.imread(path + filename, 1)
        temp_list.append(img.astype("float32"))

    temp_list = np.array(temp_list)
    #temp_list = np.reshape(temp_list,(temp_list.shape[0], temp_list.shape[1], temp_list.shape[2], 3))
    return temp_list, file_list


def load_data(path):
   
    path_train_images = path + "train/images/"
    path_train_labels = path + "train/labels/"
    path_test_images = path + "test/images/"
    path_test_labels = path + "test/labels/"
    x_train, train_image_filenames_list = load_png_images(path_train_images)
    y_train, train_label_filenames_list = load_png_images(path_train_labels)
    x_test, test_image_filenames_list = load_png_images(path_test_images)
    y_test, test_label_filenames_list = load_png_images(path_test_labels)
    x_train = normalize(x_train)
    y_train = normalize(y_train)
    x_test = normalize(x_test)
    y_test = normalize(y_test)
    return x_train, y_train, x_test, y_test, test_label_filenames_list


def load_test_images(path):
    path_test_images = path + "test/images/"
    x_test, test_image_filenames_list = load_png_images(path_test_images)
    x_test = normalize(x_test)
    return x_test, test_image_filenames_list

"""def save_results(np_array, color_space, outpath, test_label_filenames_list):
    import os
    os.makedirs(outpath, exist_ok=True)
    n = min(len(np_array), len(test_label_filenames_list))
    for i in range(n):
        pred = np_array[i]
        # Threshold for binary mask if needed
        binary_mask = (pred > 0.517).astype('uint8') * 255
        cv2.imwrite(outpath + test_label_filenames_list[i], binary_mask)"""

def save_results(np_array, color_space, outpath, test_label_filenames_list):
    os.makedirs(outpath, exist_ok=True)  # Ensure directory exists
    n = min(len(np_array), len(test_label_filenames_list))
    for i in range(n):
        pred = np_array[i]
        pred = (pred * 255).astype('uint8')  # Ensure uint8 type for saving
        cv2.imwrite(outpath + test_label_filenames_list[i], pred)
        
def save_rgb_results(np_array, outpath, test_label_filenames_list):
    i = 0
    for filename in test_label_filenames_list:
        # predict_img = np.reshape(predict_img,(predict_img[0],predict_img[1]))
        cv2.imwrite(outpath + filename, np_array[i] * 255.)
        i += 1


def save_history(model, model_name, training_history, dataset, n_filters, epoch, learning_rate, loss,
                 color_space, path=None, temp_name=None):
    save_weight_filename = temp_name if temp_name else str(datetime.datetime.now())
    model.save('{}{}.hdf5'.format(path, save_weight_filename))
    with open('{}{}.json'.format(path, save_weight_filename), 'w') as f:
        json.dump(training_history.history, f, indent=2)

    json_list = ['{}{}.json'.format(path, save_weight_filename)]
    for json_filename in json_list:
        with open(json_filename) as f:
            # convert the loss json object to a python dict
            loss_dict = json.load(f)
        print_list = ['loss', 'val_loss', 'dice_coef', 'val_dice_coef']
        for item in print_list:
            item_list = []
            if item in loss_dict:
                item_list.extend(loss_dict.get(item))
                plt.plot(item_list)
        plt.title('model:{} lr:{} epoch:{} #filtr:{} Colorspaces:{}'.format(model_name, learning_rate,
                                                                            epoch, n_filters, color_space))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'test_loss', 'train_dice', 'test_dice'], loc='upper left')
        plt.savefig('{}{}.png'.format(path, save_weight_filename))
        plt.show()
        plt.clf()



