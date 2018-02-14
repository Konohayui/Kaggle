import numpy as np
np.random.seed(123)

import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
import bisect
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate
from keras.layers import GRU, Embedding, Flatten, AveragePooling1D
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from nltk.corpus import stopwords
import re
import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FM_FTRL
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer

types_dict_train = {"train_id": "int64",
                    "item_condition_id": "int8",
                    "price": "float64",
                    "shipping": "int8"}

types_dict_test = {"test_id": "int64",
                    "item_condition_id": "int8",
                    "shipping": "int8"}

train = pd.read_csv("../input/train.tsv", sep = "\t", low_memory = True, dtype = types_dict_train)
        
train.drop(train[train.price < 3].index, inplace = True)
train = train.reset_index(drop = True)
Y_train = np.log1p(train.price)
X_train = train.drop(["price"], axis = 1)
del train
gc.collect()

def rmsle(h, y): 
    h = np.expm1(h)
    y = np.expm1(y)
    return np.sqrt(np.square(np.log1p(h) - np.log1p(y)).mean())

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, random_state = 42, test_size = 0.05)

def fill_missing_data(data):
    data.name.fillna(value = "missing", inplace = True)
    data.category_name.fillna(value = "missing/missing/missing", inplace = True)
    data.brand_name.fillna(value = "missing", inplace = True)
    data.shipping.fillna(value = 1, inplace = True)
    data.item_condition_id.fillna(value = 1, inplace = True)
    data.item_description.fillna(value = "No description yet", inplace = True)

def split_cat(data):
    data["main_cat"], data["subcat1"], data["subcat2"] = data["category_name"].str.split("/", 2).str
    data.drop(["category_name"], axis = 1, inplace = True)

def price_tag(descrip):
    words = descrip.split(" ")
    if "[rm]" in words:
        return 1
    else:
        return 0
    
def cutting(data):
    NUM_BRANDS = 4000
    NUM_CATEGORIES = 1250
    pop_brand = data["brand_name"].value_counts().loc[lambda x: x.index != "missing"].index[:NUM_BRANDS]
    data.loc[~data["brand_name"].isin(pop_brand), "brand_name"] = "missing"
    pop_category1 = data["main_cat"].value_counts().loc[lambda x: x.index != "missing"].index[:NUM_CATEGORIES]
    pop_category2 = data["subcat1"].value_counts().loc[lambda x: x.index != "missing"].index[:NUM_CATEGORIES]
    pop_category3 = data["subcat2"].value_counts().loc[lambda x: x.index != "missing"].index[:NUM_CATEGORIES]
    data.loc[~data["main_cat"].isin(pop_category1), "main_cat"] = "missing"
    data.loc[~data["subcat1"].isin(pop_category2), "subcat1"] = "missing"
    data.loc[~data["subcat2"].isin(pop_category3), "subcat2"] = "missing"
    
def to_categorical(data):
    data["main_cat"] = data["main_cat"].astype("category")
    data["subcat1"] = data["subcat1"].astype("category")
    data["subcat2"] = data["subcat2"].astype("category")
    data["item_condition_id"] = data["item_condition_id"].astype("category")
    
stopwords = {x: 1 for x in stopwords.words("english")}
non_alphanums = re.compile(u"[^A-Za-z0-9]+")
def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(" ", text).lower().strip().split(" ")] 
         if len(x) > 1 and x not in stopwords])
         
fill_missing_data(X_train)
split_cat(X_train)
X_train["product_desc"] = X_train["name"] + ": " + X_train["item_description"]
X_train["price_leak"] = X_train["product_desc"].apply(lambda x: price_tag(x))

le1 = LabelEncoder()
le1.fit(np.hstack([X_train.main_cat]))
X_train["main_catL"] = le1.transform(X_train["main_cat"])
le2 = LabelEncoder()
le2.fit(np.hstack([X_train.subcat1]))
X_train["subcat1L"] = le2.transform(X_train["subcat1"])
le3 = LabelEncoder()
le3.fit(np.hstack([X_train.subcat2]))
X_train["subcat2L"] = le3.transform(X_train["subcat2"])
le4 = LabelEncoder()
le4.fit(np.hstack([X_train.brand_name]))
X_train["brand_nameL"] = le4.transform(X_train["brand_name"])

raw_text = np.hstack([X_train.product_desc.str.lower()])
tok_raw = Tokenizer(lower = True, filters = '!"#$%&()❤️❌-*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tok_raw.fit_on_texts(raw_text)
X_train["seq_product_desc"] = tok_raw.texts_to_sequences(X_train.product_desc.str.lower())

max_desc_seq = 85
max_text = np.max([np.max(X_train.seq_product_desc.max())]) + 50
max_maincat = np.max([X_train.main_catL.max()]) + 1
max_subcat1 = np.max([X_train.subcat1L.max()]) + 1
max_subcat2 = np.max([X_train.subcat2L.max()]) + 1
max_brand_name = np.max([X_train.brand_nameL.max()]) + 1
max_condition = np.max([X_train.item_condition_id.max()]) + 1

def get_keras_data(data):
    df = {"product_desc": pad_sequences(data.seq_product_desc, maxlen = max_desc_seq),
          "brand_name": np.array(data.brand_nameL),
          "main_cat": np.array(data.main_catL),
          "subcat1": np.array(data.subcat1L),
          "subcat2": np.array(data.subcat2L),
          "item_condition": np.array(data.item_condition_id),
          "shipping": np.array(data[["shipping"]]),
          "price_tag": np.array(data[["price_leak"]])}
    return df

X_train_K = get_keras_data(X_train)

def build_model(lr_i = 0.0, lr_f = 0.0, batch_size = 0):
    product_desc = Input(shape = [X_train_K["product_desc"].shape[1]], name = "product_desc")
    brand_name = Input(shape = [1], name = "brand_name")
    main_cat = Input(shape = [1], name = "main_cat")
    subcat1 = Input(shape = [1], name = "subcat1")
    subcat2 = Input(shape = [1], name = "subcat2")
    item_condition = Input(shape = [1], name = "item_condition")
    shipping = Input(shape = [X_train_K["shipping"].shape[1]], name = "shipping")
    price_tag = Input(shape = [X_train_K["price_tag"].shape[1]], name = "price_tag")

    emb_size = 30
    emb_product_desc = Embedding(max_text, emb_size)(product_desc)
    emb_brand_name = Embedding(max_brand_name, 10)(brand_name)
    emb_main_cat = Embedding(max_maincat, 10)(main_cat)
    emb_subcat1 = Embedding(max_subcat1, 10)(subcat1)
    emb_subcat2 = Embedding(max_subcat2, 10)(subcat2)
    emb_item_condition = Embedding(max_condition, 5)(item_condition)
    
    pool_layer = AveragePooling1D()(emb_product_desc)
    rnn_layer = GRU(25)(pool_layer)

    main_layer = concatenate([Flatten()(emb_brand_name), 
                              Flatten()(emb_item_condition), 
                              Flatten()(emb_main_cat),
                              Flatten()(emb_subcat1),
                              Flatten()(emb_subcat2),
                              rnn_layer, 
                              shipping,
                              price_tag])
    
    main_layer = Dense(128, activation = "relu") (main_layer)
    main_layer = Dense(64, activation = "relu") (main_layer)
    main_layer = Dense(32, activation = "relu") (main_layer)

    out_put = Dense(1, activation = "linear")(main_layer)
    model = Model(inputs = [product_desc, brand_name, main_cat, subcat1,
                            subcat2, item_condition, shipping, price_tag], outputs = out_put)

######################################################################
#
    steps = np.multiply(int(len(X_train["shipping"])/batch_size), 2)
    lr_init, lr_fin = lr_i, lr_f 
    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    lr_decay = exp_decay(lr_init, lr_fin, steps)
#
######################################################################

    model.compile(loss = "mse", optimizer = Adam(lr = lr_init, decay = lr_decay), metrics = ["mae"])
    return model

batch_size = 1024
K_model = build_model(lr_i = 0.009, lr_f = 0.001, batch_size = batch_size)
K_model.fit(X_train_K, Y_train, epochs = 4, batch_size = batch_size, verbose = 10)
print("Keras model training completed!")

X_train.drop(["main_catL", "subcat1L", "subcat2L", "brand_nameL", "seq_product_desc", "price_leak"], axis = 1)
del X_train_K, raw_text
gc.collect()

cutting(X_train)
to_categorical(X_train)

wb1 = wordbatch.WordBatch(normalize_text, \
                        extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                             "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                             "idf": None,}), procs = 8)
wb1.dictionary_freeze = True
wb1.fit(X_train["name"])
X_train_name = wb1.transform(X_train["name"])
mask1 = np.where(X_train_name.getnnz(axis = 0) > 1)[0]
X_train_name = X_train_name[:, mask1]

wb2 = wordbatch.WordBatch(normalize_text, \
                        extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                             "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                             "idf": None}), procs = 8)
wb2.dictionary_freeze = True
wb2.fit(X_train["item_description"])
X_train_description = wb2.transform(X_train["item_description"])
mask2 = np.where(X_train_description.getnnz(axis = 0) > 1)[0]
X_train_description = X_train_description[:, mask2]

cv1 = CountVectorizer()
cv1.fit(X_train["main_cat"])
X_train_category1 = cv1.transform(X_train["main_cat"])

cv2 = CountVectorizer()
cv2.fit(X_train["subcat1"])
X_train_category2 = cv2.transform(X_train["subcat1"])

cv3 = CountVectorizer()
cv3.fit(X_train["subcat2"])
X_train_category3 = cv3.transform(X_train["subcat2"])

lb = LabelBinarizer(sparse_output = True)
lb.fit(X_train["brand_name"])
X_train_brand = lb.transform(X_train["brand_name"])

X_train_dummies = csr_matrix(pd.get_dummies(X_train[["item_condition_id", "shipping"]], sparse = True).values)

X_train = hstack((X_train_dummies, X_train_description, X_train_brand, 
                  X_train_category1, X_train_category2, X_train_category3, X_train_name)).tocsr()
mask3 = np.where(X_train.getnnz(axis = 0) > 1)[0]
X_train = X_train[:, mask3]
d = X_train.shape[1]

del X_train_dummies, X_train_description, X_train_brand, X_train_category1, X_train_category2, X_train_category3, X_train_name
gc.collect()

FM_model = FM_FTRL(alpha = 0.01, beta = 0.01, L1 = 0.00001, L2 = 0.1, D = d, 
                alpha_fm = 0.01, L2_fm = 0.0, init_fm = 0.01, D_fm = 200, e_noise = 0.0001, 
                iters = 15, inv_link = "identity", threads = 4)
FM_model.fit(X_train, Y_train)
print("FM_FTRL training completed!")

del d, X_train, Y_train
gc.collect()

def prediction_process(test, le1 = le1, le2 = le2, le3 = le3, le4 = le4, tok_raw = tok_raw, batch_size = batch_size,
                       wb1 = wb1, wb2 = wb2, mask1 = mask1, mask2 = mask2, mask3 = mask3, lb = lb,
                       K_model = K_model, FM_model = FM_model):
    
    fill_missing_data(test)
    split_cat(test)
    test["product_desc"] = test["name"] + ": " + test["item_description"]
    test["price_leak"] = test["product_desc"].apply(lambda x: price_tag(x))
    
    le_classes = le1.classes_.tolist()
    bisect.insort_left(le_classes, "missing")
    le1.classes_ = le_classes
    test["main_cat"] = test["main_cat"].map(lambda x: "missing" if x not in le1.classes_ else x)
    test["main_catL"] = le1.transform(test["main_cat"])
    
    le_classes = le2.classes_.tolist()
    bisect.insort_left(le_classes, "missing")
    le2.classes_ = le_classes
    test["subcat1"] = test["subcat1"].map(lambda x: "missing" if x not in le2.classes_ else x)
    test["subcat1L"] = le2.transform(test["subcat1"])
    
    le_classes = le3.classes_.tolist()
    bisect.insort_left(le_classes, "missing")
    le3.classes_ = le_classes
    test["subcat2"] = test["subcat2"].map(lambda x: "missing" if x not in le3.classes_ else x)
    test["subcat2L"] = le3.transform(test["subcat2"])
    
    le_classes = le4.classes_.tolist()
    bisect.insort_left(le_classes, "missing")
    le4.classes_ = le_classes
    test["brand_name"] = test["brand_name"].map(lambda x: "missing" if x not in le4.classes_ else x)
    test["brand_nameL"] = le4.transform(test["brand_name"])
    
    test["seq_product_desc"] = tok_raw.texts_to_sequences(np.hstack([test.product_desc.str.lower()]))
    X_test_K = get_keras_data(test)
    
    K_pred = K_model.predict(X_test_K, batch_size = batch_size, verbose = 10)
    test.drop(["main_catL", "subcat1L", "subcat2L", "brand_nameL", "seq_product_desc", "price_leak"], axis = 1)
    del X_test_K
    gc.collect()
    
    cutting(test)
    to_categorical(test)
    X_test_name = wb1.transform(test["name"])
    X_test_name = X_test_name[:, mask1]
    X_test_description = wb2.transform(test["item_description"])
    X_test_description = X_test_description[:, mask2]
    
    X_test_category1 = cv1.transform(test["main_cat"])
    X_test_category2 = cv2.transform(test["subcat1"])
    X_test_category3 = cv3.transform(test["subcat2"])
    X_test_brand = lb.transform(test["brand_name"])
    X_test_dummies = csr_matrix(pd.get_dummies(test[["item_condition_id", "shipping"]], sparse = True).values)
    
    X_test = hstack((X_test_dummies, X_test_description, X_test_brand, 
                     X_test_category1, X_test_category2, X_test_category3, X_test_name)).tocsr()
    X_test = X_test[:, mask3]
    
    del X_test_dummies, X_test_description, X_test_brand, X_test_category1, X_test_category2, X_test_category3, X_test_name
    gc.collect()
    del test
    gc.collect()
    
    FM_pred = FM_model.predict(X_test)
    pred = np.expm1(0.4*K_pred[:, 0] + 0.6*FM_pred)
    
    del X_test, K_pred, FM_pred
    gc.collect()
    return pred

def load_test():
    for df in pd.read_csv("../input/test.tsv", sep = "\t", low_memory = True, dtype = types_dict_test, chunksize = 700000):
        yield df

print("Prediction, start!")
preds = []
for test in load_test():
    pred = prediction_process(test)
    preds.append(pred)

submission = pd.read_table("../input/test.tsv", engine = "c", usecols = ["test_id"])  
submission["price"] = pd.DataFrame(np.concatenate(preds, axis = 0))
submission.to_csv("submission.csv", index = False)
