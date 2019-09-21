import json
import pandas as pd
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
import numpy as np
import random

review_word_num = 50  # the number of words per reivew
max_word_num = 200000  # the maximun words in vocabulary

u_review_num = 5 # the number of review per user
i_review_num = 10 # the number of review per item

DataSet_Name = 'Automotive_5'
Input_DataSet_File = './data/'+DataSet_Name + '/' + DataSet_Name + '.json'
Output_PreprocessData_Path = './data/'+DataSet_Name + '/'


print('Input_DataSet_File:',Input_DataSet_File)

'''
    load the raw data and filter out the useless characters.
'''
raw_data = [json.loads(i) for i in open(Input_DataSet_File, "rt")]

data = pd.DataFrame(raw_data).loc[:,
                                  ["reviewerID",
                                   "reviewText",
                                   "asin",
                                   "overall"]
                                ]
cleaned_text = data.loc[:, ["reviewerID", "asin", "overall"]]

# filter out the useless characters.
def clean(text):
    return text_to_word_sequence(text,
                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                 lower=True, split=" ")

#  the every single line of data.loc[:, "reviewText"] is filtered by function 'def clean(text)'
cleaned_text.loc[:, "reviewText"] = data.loc[:, "reviewText"].apply(clean)
# ui_id denotes the ID of user-item pair 
cleaned_text["ui_id"]=np.array([ i for i in range(cleaned_text.shape[0])]).reshape(-1,1) 



"""
    Tokenize the reviews，and padding 0 or 
    truncate based on MAX_SEQUENCE_LENGTH
"""
# 向量化文本样本
tokenizer_reviews = Tokenizer(num_words=max_word_num)
# fit_on_text(texts) 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档。就是对文本单词进行去重后
tokenizer_reviews.fit_on_texts(cleaned_text['reviewText'])
# texts_to_sequences(texts) 将多个文档转换为word在词典中索引的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)
token_reviews = tokenizer_reviews.texts_to_sequences(cleaned_text['reviewText'])
# 长度超过MAX_SEQUENCE_LENGTH的评论则截断，不足则补0
padding_reviews = pad_sequences(token_reviews, maxlen=review_word_num)
print('the total number of words in the whole datasets is:', len(tokenizer_reviews.word_index))

"""
    Tokenize users ID
"""
tokenizer_user =  Tokenizer(num_words=None)
tokenizer_user.fit_on_texts(cleaned_text['reviewerID'])
token_userID = tokenizer_user.texts_to_sequences(cleaned_text['reviewerID'])
print('the total number of users is:', len(tokenizer_user.word_index))

"""
    Tokenize items ID
"""
tokenizer_item =  Tokenizer(num_words=None)
tokenizer_item.fit_on_texts(cleaned_text['asin'])
token_itemID = tokenizer_item.texts_to_sequences(cleaned_text['asin'])
print('the total number of items is:', len(tokenizer_item.word_index))

'''
    Contruct the dataFrame of tokenized data
'''
processed_data = {
    "reviewerID" : list(map(lambda x : x[0],token_userID)),
    "asin" : list(map(lambda x : x[0],token_itemID)),
    "overall": cleaned_text['overall'],
    "ui_id": cleaned_text['ui_id'],
    "reviewText":padding_reviews.tolist()
   }
token_data = pd.DataFrame(processed_data)#将字典转换成为数据框
#重新排列token_data的列顺序
cols=['reviewerID','asin','overall','reviewText','ui_id']
token_data=token_data.loc[:,cols]


'''
    Divide the tokenized dataset into trainSet, validSet, testSet
'''
shuffle_token_data = token_data.loc[np.random.permutation(np.arange(len(token_data)))]
train_size = int(len(token_data)*0.8)
valid_size = int(len(token_data)*0.1)
test_size = int(len(token_data)*0.1)
train = shuffle_token_data[0:train_size]
valid = shuffle_token_data[train_size:(train_size+valid_size)]
test = shuffle_token_data[-test_size:]


# construct the user/item reivew set
'''
    param:
    user_reviews: 
            a dictionary, whose key is user_id and 
                     the value is a list consists of all the reivews written by this user.
    ur_iid: 
            a dictionary, whose key is a user_id and the value is a list consists of
                     the item_id correspond to his reviews.
'''
user_reviews={}
item_reviews={}
ur_iid={}
ir_uid={}

'''
    the train reviews can been seen by the model, 
''' 
for index, row in train.iterrows():
    # process user reviews
    if row['reviewerID'] in user_reviews:
        user_reviews[row['reviewerID']].append(row['reviewText'])
        ur_iid[row['reviewerID']].append(row['asin'])
    else:
        user_reviews[row['reviewerID']] = [row['reviewText']]
        ur_iid[row['reviewerID']] = [row['asin']]
        
    # process item reviews
    if row['asin'] in item_reviews:
        item_reviews[row['asin']].append(row['reviewText'])
        ir_uid[row['asin']].append(row['reviewerID'])
    else:
        item_reviews[row['asin']] = [row['reviewText']]
        ir_uid[row['asin']] = [row['reviewerID']]
                    
'''
    the valid/test reviews can not been seen by the model, and
    we set these reviews as NULL by '0'
'''
# 1. process valid
for  index, row in valid.iterrows():
    # process user reviews
    if row['reviewerID'] in user_reviews:
        a=1 # do nothing
    else:
        # this user_id is not in train, and we set his review as NULL
        user_reviews[row['reviewerID']] = [[int(0) for _ in range(review_word_num)]]
        ur_iid[row['reviewerID']] = [0]
        
    # process item reviews
    if row['asin'] in item_reviews:
        a=1 # do nothing
    else:
        # this item_id is not in train, and we set his review as NULL
        item_reviews[row['asin']] = [[int(0) for _ in range(review_word_num)]]
        ir_uid[row['asin']] = [0]
        
# 2. process test
for  index, row in test.iterrows():
    # process user reviews
    if row['reviewerID'] in user_reviews:
        a=1 # do nothing
    else:
        # this user_id is not in train, and we set his review as NULL
        user_reviews[row['reviewerID']] = [[int(0) for _ in range(review_word_num)]]
        ur_iid[row['reviewerID']] = [0]
        
    # process item reviews
    if row['asin'] in item_reviews:
        a=1 # do nothing
    else:
        # this item_id is not in train, and we set his review as NULL
        item_reviews[row['asin']] = [[int(0) for _ in range(review_word_num)]]
        ir_uid[row['asin']] = [0]



'''
    Pack the core parameters as a dictionary called 'param',
    and store.
'''
# Pack the core parameters as 'param'
param={}
param['review_word_num'] = review_word_num
param['user_num'] = len(tokenizer_user.word_index) # the number of users of whole dataset
param['item_num'] = len(tokenizer_item.word_index) # the number of items of whole dataset
param['vocabulary'] = tokenizer_reviews.word_index # vocabulary
param['train'] = train # type(train)=pandas dataFrame
param['valid'] = valid # type(valid)=pandas dataFrame
param['test'] = test # type(test)=pandas dataFrame
param['user_reviews'] = user_reviews
param['ur_iid'] = ur_iid
param['item_reviews'] = item_reviews
param['ir_uid'] = ir_uid
param['u_review_num'] = u_review_num
param['i_review_num'] = i_review_num

# store 'param' in hard disk
pickle.dump(param, open(Output_PreprocessData_Path+'param.dict', 'wb'))

print('Data Preprocess Completed !')