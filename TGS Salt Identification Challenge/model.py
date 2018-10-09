import numpy as np, pandas as pd, random as rn, time, gc, os
np.random.seed(32)

from random import randint

from sklearn.model_selection import train_test_split
from skimage.transform import resize

import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'

from keras import backend as K
rn.seed(32)
tf.set_random_seed(32)
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

from keras.preprocessing.image import load_img
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import Model

from tqdm import tqdm_notebook

train = pd.read_csv("../input/train.csv")
depth = pd.read_csv("../input/depths.csv")
train = pd.merge(train, depth, on = "id")

ori_image_size = 101
tar_image_size = 128

def up_sample_size(img):
    if ori_image_size == tar_image_size:
        return img
    return resize(img, (tar_image_size, tar_image_size), mode = "constant", preserve_range = True)

def down_sample_size(img):
    if ori_image_size == tar_image_size:
        return img
    return resize(img, (ori_image_size, ori_image_size), mode = "constant", preserve_range = True)
train["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train["id"])]
train["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train["id"])]
def cov_to_classes(val):
    for i in range(0, 11):
        if val*10 <= i:
            return i

train["coverage"] = train["masks"].map(np.sum)/pow(ori_image_size, 2)
train["coverage_class"] = train["coverage"].apply(lambda x: cov_to_classes(x))
# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
def double_conv_layer(x, size, dr = 0.0, batch_norm = True):
    conv = Conv2D(size, (3, 3), padding = "same")(x)
    if batch_norm is True:
        conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(size, (3, 3), padding = "same")(conv)
    if batch_norm is True:
        conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    if dr > 0.0:
        conv = SpatialDropout2D(dr)(conv)
    
    return conv

def build_model(lr = 1e-3, de = 0.0, dr = 0.0):
    inputs = Input((tar_image_size, tar_image_size, 1))
    
    c128 = double_conv_layer(inputs, 16, dr = dr, batch_norm = True)
    pool_64 = MaxPooling2D(pool_size = (2, 2))(c128)
    
    c64 = double_conv_layer(pool_64, 32, dr = dr, batch_norm = True)
    pool_32 = MaxPooling2D(pool_size = (2, 2))(c64)
    
    c32 = double_conv_layer(pool_32, 64, dr = dr, batch_norm = True)
    pool_16 = MaxPooling2D(pool_size = (2, 2))(c32)
    
    c16 = double_conv_layer(pool_16, 128, dr = dr, batch_norm = True)
    pool_8 = MaxPooling2D(pool_size = (2, 2))(c16)
    
    c8 = double_conv_layer(pool_8, 256, dr = dr, batch_norm = True)
    pool_8 = MaxPooling2D(pool_size = (2, 2))(c8)
    
    c4 = double_conv_layer(pool_8, 512, dr = dr, batch_norm = True)
    
    u8 = concatenate([UpSampling2D(size = (2, 2))(c4), c8])
    uc8 = double_conv_layer(u8, 256, dr = dr, batch_norm = True)
    
    u16 = concatenate([UpSampling2D(size = (2, 2))(uc8), c16])
    uc16 = double_conv_layer(u16, 128, dr = dr, batch_norm = True)
    
    u32 = concatenate([UpSampling2D(size = (2, 2))(uc16), c32])
    uc32 = double_conv_layer(u32, 64, dr = dr, batch_norm = True)
    
    u64 = concatenate([UpSampling2D(size = (2, 2))(uc32), c64])
    uc64 = double_conv_layer(u64, 32, dr = dr, batch_norm = True)
    
    u128 = concatenate([UpSampling2D(size = (2, 2))(uc64), c128])
    uc128 = double_conv_layer(u128, 16, dr = dr, batch_norm = True)
    
    outputs = Conv2D(1, (1, 1), activation = "sigmoid", padding = "same")(uc128)
    model = Model(inputs = [inputs], outputs = [outputs])
    model.compile(loss = bce_dice_loss, optimizer = Adam(lr = lr, decay = de),
                  metrics = ["accuracy"])
    
    return model
    
    
from sklearn.model_selection import KFold

n_fold = 5
oof_preds = np.zeros((train.shape[0], tar_image_size, tar_image_size, 1), np.uint8)
train.sort_values("z", inplace = True)
train["fold"] = (list(range(5))*train.shape[0])[:train.shape[0]]
batch_size = 32
epochs = 100
lr = 1e-3
dr = 0.25
do_Kfold = True

if do_Kfold:
    for fold in range(n_fold):
        print("="*50)
        print("\n Training {}/{} fold...".format(fold+1, n_fold))
        fold_train = train[train["fold"] != fold] 
        fold_valid = train[train["fold"] == fold]
        
        X_train = np.array(fold_train["images"].map(up_sample_size).tolist()).reshape(-1, tar_image_size, tar_image_size, 1)
        Y_train = np.array(fold_train["masks"].map(up_sample_size).tolist()).reshape(-1, tar_image_size, tar_image_size, 1)
        
        X_valid = np.array(fold_valid["images"].map(up_sample_size).tolist()).reshape(-1, tar_image_size, tar_image_size, 1)
        Y_valid = np.array(fold_valid["masks"].map(up_sample_size).tolist()).reshape(-1, tar_image_size, tar_image_size, 1)
        
        # image augmentaion
        X_train = np.append(X_train, [np.fliplr(x) for x in X_train], axis = 0)
        Y_train = np.append(Y_train, [np.fliplr(x) for x in Y_train], axis = 0)

        file_path = "fold_{}_model.hdf5".format(fold+1)
        early_stop = EarlyStopping(monitor = "val_loss", patience = 10, 
                                   min_delta = 1e-4, verbose = 1, mode = "min")
        check_point = ModelCheckpoint(file_path, save_best_only = True, verbose = 1)
        lr_schedule = ReduceLROnPlateau(minitor = "val_loss", factor = 0.5, 
                                        patience = 5, min_lr = 1e-6, verbose = 1, 
                                        epsilom = 1e-4, mode = "min")

        model = build_model(lr = lr, de = 0.0, dr = dr)
        history = model.fit(X_train, Y_train,
                            validation_data = (X_valid, Y_valid),
                            epochs = epochs,
                            batch_size = batch_size,
                            callbacks = [early_stop, check_point, lr_schedule],
                            verbose = 10)
        
        model = load_model(file_path, custom_objects = {"bce_dice_loss": bce_dice_loss})
        oof_preds[train["fold"] == fold] = model.predict(X_valid, verbose = 2).reshape(-1, tar_image_size, tar_image_size, 1)
        
else:
    train_idx, valid_idx, X_train, X_valid, Y_train, Y_valid = train_test_split(train["id"],
            np.array(train["images"].map(up_sample_size).tolist()).reshape(-1, tar_image_size, tar_image_size, 1),
            np.array(train["masks"].map(up_sample_size).tolist()).reshape(-1, tar_image_size, tar_image_size, 1),
            test_size = 0.2, stratify = train.coverage_class, random_state = 32)
    
    
    X_train = np.append(X_train, [np.fliplr(x) for x in X_train], axis = 0)
    Y_train = np.append(Y_train, [np.fliplr(x) for x in Y_train], axis = 0)
        
    file_path = "model.hdf5"
    early_stop = EarlyStopping(monitor = "val_loss", patience = 10, 
                               min_delta = 1e-4, verbose = 1, mode = "min")
    check_point = ModelCheckpoint(file_path, save_best_only = True, verbose = 1)
    lr_schedule = ReduceLROnPlateau(minitor = "val_loss", factor = 0.5, 
                                    patience = 5, min_lr = 1e-6, verbose = 1, 
                                    epsilom = 1e-4, mode = "min")

    model = build_model(lr = lr, de = 0.0, dr = dr)
    history = model.fit(X_train, Y_train,
                        validation_data = (X_valid, Y_valid),
                        epochs = epochs,
                        batch_size = batch_size,
                        callbacks = [early_stop, check_point, lr_schedule],
                        verbose = 10)
                        
                        
import matplotlib.pyplot as plt
import seaborn as sns

if not do_Kfold:
    plt.figure(figsize = (15, 5))
    plt.plot(history.epoch, history.history["loss"], label = "Train Loss")
    plt.plot(history.epoch, history.history["val_loss"], label = "Validation Loss")
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.0)
    plt.xlabel("epoch")
    plt.ylabel("dice loss")
    plt.show()
if not do_Kfold:
    plt.figure(figsize = (15, 5))
    plt.plot(history.epoch, history.history["acc"], label = "Train Pixel Accuracy")
    plt.plot(history.epoch, history.history["val_acc"], label = "Validation Pixel Accuracy")
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.0)
    plt.xlabel("epoch")
    plt.ylabel("pixel accuracy")
    plt.show()
if not do_Kfold:
    model = load_model(file_path, custom_objects = {"bce_dice_loss": bce_dice_loss})
    valid_preds = model.predict(X_valid).reshape(-1, tar_image_size, tar_image_size, 1)
    valid_preds = np.array([down_sample_size(pre) for pre in valid_preds])
    valid_ori = np.array([down_sample_size(ori) for ori in Y_valid])

else:
    valid_preds = np.array([down_sample_size(pre) for pre in oof_preds])
    valid_ori = np.array([down_sample_size(ori) for ori in train["masks"]])
    
thresholds = np.linspace(0, 1, 50)
ious = np.array([iou_metric_batch(valid_ori, np.int32(valid_preds > threshold)) for threshold in tqdm_notebook(thresholds)])

threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
print("Best Threshold: {}, Best IoU: {}".format(threshold_best, iou_best))

test_idx = next(os.walk("../input/test/images"))[2]
X_test = np.array([up_sample_size(np.array(load_img("../input/test/images/{}".format(idx), grayscale = True)))/255 for idx in tqdm_notebook(test_idx)]).reshape(-1, tar_image_size, tar_image_size, 1)

if not do_Kfold:
    X_pred = model.predict(X_test, verbose = 1)
else:
    X_pred = 0
    for fold in range(n_fold):
        model = load_model("fold_{}_model.hdf5".format(fold+1), custom_objects = {"bce_dice_loss": bce_dice_loss})
        X_pred += model.predict(X_test, verbose = 1)/n_fold
        
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs
        
        
predictions = {idx[:-4]:RLenc(np.round(down_sample_size(X_pred[i]))) for i, idx in tqdm_notebook(enumerate(test_idx))}
submission = pd.DataFrame.from_dict(predictions, orient = "index")
submission.index.names = ["id"]
submission.columns = ["rle_mask"]
submission.to_csv("submission.csv")
        
        

