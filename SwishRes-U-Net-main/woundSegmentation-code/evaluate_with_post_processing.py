import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm
from utils.io.data import get_png_filename_list
from utils.postprocessing.hole_filling import fill_holes
from utils.postprocessing.remove_small_noise import remove_small_areas

#save_model_name = "model-DUnet-Fuseg-aug-100ep"
save_model_name = "model-DUnet-AZHData-aug-100ep"

#Train on Fuseg,Test on Fuseg
#path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/results-SOTA/Fuseg/crossCorpora/' + save_model_name + '/' #pred
#label_path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/data/FuSeg/validation/labels/' #ground truth labels

#Train on Fuseg,Test on AZH
#path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/results-SOTA/azh/crossCorpora/' + save_model_name + '/' #pred
#label_path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/data/azh/test/labels/' #ground truth labels

#Train on AZH,Test on AZH
#path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/results-SOTA/azh/crossCorpora/' + save_model_name + '/' #pred
#label_path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/data/azh/test/labels/' #ground truth labels

#Train on AZH,Test on Fuseg
path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/results-SOTA/Fuseg/crossCorpora/' + save_model_name + '/' #pred
label_path = 'C:/Users/kwame/OneDrive/Desktop/UNI STUFF/COMPUTER SCIENCE/L400/Project Work/Test/data/FuSeg/validation/labels/' #ground truth labels

def evaluate(threshold, file_list, label_path, post_prosecced_path):
    false_positives = 0
    false_negatives = 0
    true_positives = 0

    # for img_name in tqdm(file_list):
    #     img = cv2.imread(pred_dir + img_name)
    #     _, threshed = cv2.threshold(img, threshold, 255, type=cv2.THRESH_BINARY)
    #     ###############################################################################################################
    #     # call image post processing functions
    #     mask = np.zeros((226, 226, 3))
    #     filled = fill_holes(threshed, threshold,0.1)
    #     denoised = remove_small_areas(filled, threshold, 0.05)
    #     ################################################################################################################
    #     cv2.imwrite(path + 'filled/' + img_name, filled)
    #     cv2.imwrite(path + 'postProcessed/' + img_name, denoised)
        # cv2.imwrite(path + 'bMask/' + img_name, threshed)


    for filename in tqdm(file_list):
        label = cv2.imread(label_path + filename,0)
        label = cv2.resize(label, (224,224))
        post_prosecced = cv2.imread(post_prosecced_path + filename,0)
        xdim = label.shape[0]
        ydim = label.shape[1]
        # print (xdim,ydim)
        for x in range(xdim):
            for y in range(ydim):
                # print(post_prosecced[x, y-1],label[x, y-1])
                if post_prosecced[x, y] and label[x, y] > threshold:
                    true_positives += 1
                if label[x, y] > threshold > post_prosecced[x, y]:
                    false_negatives += 1
                if label[x, y] < threshold < post_prosecced[x, y]:
                    false_positives += 1

    IOU = float(true_positives) / (true_positives + false_negatives + false_positives)
    Dice = 2*float(true_positives) / (2*true_positives + false_negatives + false_positives)
    precision = float(true_positives) / (true_positives + false_positives)
    recall = float(true_positives) / (true_positives + false_negatives)

    print("--------------------------------------------------------")
    print("Weight file: ",post_prosecced_path.rsplit("/")[1])
    print("--------------------------------------------------------")
    print("Threshold: ", threshold)
    print("True  pos = " + str(true_positives))
    print("False neg = " + str(false_negatives))
    print("False pos = " + str(false_positives))
    print("IOU = " + str(IOU))
    print("Dice = " + str(Dice))
    print("Precision = " + str(precision))
    print("Recall = " + str(recall))


# change to your own folder names
pred_dir = path + 'model-pred-masks/'
img_filename_list = get_png_filename_list(pred_dir)

num_threads = multiprocessing.cpu_count()
# test your own threshold
threshold = 100
print("Predcted Masks")
evaluate(threshold, img_filename_list, label_path, pred_dir)
