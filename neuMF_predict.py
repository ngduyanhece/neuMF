import numpy as np

import keras
from keras import backend as K
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import GMF, MLP
import argparse
import json
from sklearn.utils import shuffle
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='vtc_cab',
                        help='Choose a dataset.')
    return parser.parse_args()


def get_model(num_users, num_items, latent_dim, regs=[0, 0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MF_Embedding_User = Embedding(
        input_dim=num_users,
        output_dim=latent_dim,
        name='user_embedding',
        embeddings_initializer='normal',
        embeddings_regularizer=l2(regs[0]),
        input_length=1)
    MF_Embedding_Item = Embedding(
        input_dim=num_items,
        output_dim=latent_dim,
        name='item_embedding',
        embeddings_initializer='normal',
        embeddings_regularizer=l2(regs[1]),
        input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))

    # Element-wise product of user and item embeddings
    predict_vector = merge([user_latent, item_latent], mode='mul')

    # Final prediction layer
    # prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)

    model = Model(input=[user_input, item_input],
                  output=prediction)

    return model

if __name__ == '__main__':
    args = parse_args()
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    rating = testRatings[0]
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))
    idx_account = json.load(open('./Data/idx_account.json'))
    idx_key = json.load(open('./Data/idx_key.json'))
    idx_account = {int(k): str(v) for k, v in idx_account.items()}
    idx_key = {int(k): str(v) for k, v in idx_key.items()}
    model = get_model(num_users, num_items, 20, [1e-5,1e-5])
    model.compile(optimizer=RMSprop(lr=0.000001), loss='binary_crossentropy')
    print(model.summary())
    model.load_weights('./Pretrain/vtc_cab_GMF_20_1518366491.h5')
    #sample make prediction for 5 customer
    items = list(idx_key.keys())
    uids = shuffle(list(idx_account.keys()))[0:10]
    purchases = pd.read_csv('./Data/retail_group.csv')
    print(purchases.head(10))
    for cus in uids:
        users = np.full(len(items), cus, dtype='int32')
        predictions = model.predict([users, np.array(items)])
        top_ten = predictions.reshape(-1, ).argsort()[-10:][::-1]
        print("already purchased items")
        print(purchases[purchases['accountid'] == str(idx_account[cus])]['key'])
        print("item recommended for customer {}".format(idx_account[cus]))
        for item in top_ten:
            print(idx_key[item])
        print("_______________________________________")