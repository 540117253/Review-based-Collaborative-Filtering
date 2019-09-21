import numpy as np
import tensorflow as tf
import math
from tensorflow.contrib import learn
from datetime import datetime 
import pickle
import random
import os
import gc
from models.wordVectorBased_Models import DARR, DeepCoNN, MACF




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 屏蔽TensorFlow的通知信息和警告信息


# 数据集名称
Dataset_Name = 'Automotive_5'
# 处理好的数据集路径
DIR = './data/' + Dataset_Name + '/'
# 预训练词向量文件路径
GLOVE_DIR = './data/' + 'glove.6B'



# DataSet Path
tf.flags.DEFINE_string("para_data", DIR + "param.dict", "Data parameters and preprocessed data")

# Model Shared Hyperparameters
tf.flags.DEFINE_boolean("word2vec", True, "it denotes whether use the pre-trained word vector")
tf.flags.DEFINE_integer("word_emb_size", 50, "the dimension of word vector")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "dropout keep probability")
tf.flags.DEFINE_float("init_lr", 0.0005, "initial learing rate, default 0.001") 
tf.flags.DEFINE_integer("n_latent", 20 , "the number of lantent factors in Deep Model")
tf.flags.DEFINE_integer("fm_k", 6 , "the dimension of factorization machines")
tf.flags.DEFINE_integer("rnn_output_dim", 100 , "the output hidden state dimension of the RNN module")
tf.flags.DEFINE_integer("attention_dim", 50 , "the dimension of the attention layer")
tf.flags.DEFINE_string("filter_sizes", "3" , "CNN filter size")
tf.flags.DEFINE_integer("num_filters", 30 , "CNN filter num")

# Training parameters
tf.flags.DEFINE_integer("batch_size",128, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs ")

# GPU Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 0.8, "the maximum memory that Model can use in one GPU")

'''
    函数def generate_batch(batch_num, batch_size, data_size_train, shuffled_data , batch_data):
    
    功能：
        根据batch编号batch_num及batch_size，从shuffled_data中生成相应的batch数据
    
    变量：
        batch_num：batch的编号
        batch_size：每个batch的大小
        data_size：训练集的总长度
        input_data： 输入的数据集
        batch_u_text：当前batch中用户对应的评论集
        batch_i_text：当前batch中商品对应的评论集
        batch_uid：当前batch中用户的id集
        batch_iid：当前batch中商品的id集
        batch_y：当前batch中的评分集
'''
def generate_batch(batch_index, batch_size, data_size, input_data):
    start_index = batch_index * batch_size # 该batch 在 shuffled_data 中的起始位置索引
    end_index = min((batch_index + 1) * batch_size, data_size) # 该batch 在 shuffled_data 中的结束位置索引
    
    batch_uid = np.reshape(input_data[start_index:end_index]['reviewerID'].values, [-1,1]) # shape = [batch_size,1]
    batch_iid = np.reshape(input_data[start_index:end_index]['asin'].values, [-1,1]) # shape = [batch_size,1]
    batch_y = np.reshape(input_data[start_index:end_index]['overall'].values, [-1,1]) # shape = [batch_size,1]

    batch_u_text = [] # after the following process, its shape=[batch_size,u_review_num,review_word_num]
    batch_i_text = [] # after the following process, its shape=[batch_size,i_review_num,review_word_num]
    for i in range(len(batch_iid)):
        # current user ID
        cur_uid = batch_uid[i][0]
        # current item ID
        cur_iid = batch_iid[i][0]
        
        '''
            process the current user's review set
        '''
        try: # get the reivew index of the current item ID in user_reivew set
            ur_index = ur_iid[cur_uid].index(cur_iid)
        except: 
            ur_index = -1
        cur_u_text = []
        # the review of 'cur_iid' can not been seen by the model when predict the rating of current item
        for j in range(len(ur_iid[cur_uid])):
            if j != ur_index:
                cur_u_text.append(user_reviews[cur_uid][j])
        # truncate the length by randomly selecting 'u_review_num' reviews
        if len(cur_u_text)>u_review_num: 
            random.shuffle(cur_u_text)
            cur_u_text=cur_u_text[0:u_review_num]
        else: # pad the length of current user'review set until it is equal to u_review_num
            while len(cur_u_text) < u_review_num: 
                cur_u_text.append([int(0) for _ in range(review_word_num)])
        # add the current user's review set to batch_u_text
        batch_u_text.append(cur_u_text)
            
        '''
            process the current item's review set
        '''    
        # get the review index of the current user ID in item_reivew set
        try:
            ir_index = ir_uid[cur_iid].index(cur_uid)
        except:
            ir_index = -1
        cur_i_text = []
        for j in range(len(ir_uid[cur_iid])):
            if j != ir_index:
                cur_i_text.append(item_reviews[cur_iid][j])
        # truncate the length by randomly selecting 'i_review_num' reviews
        if len(cur_i_text)>i_review_num: 
            random.shuffle(cur_i_text)
            cur_i_text=cur_i_text[0:i_review_num]
        else: # pad the length of current item'review set until it is equal to i_review_num    
            while len(cur_i_text) < i_review_num:  
                cur_i_text.append([int(0) for _ in range(review_word_num)])
        # add the current item's review set to batch_i_text
        batch_i_text.append(cur_i_text)

    batch_u_text = np.array(batch_u_text)
    batch_i_text = np.array(batch_i_text)
    
    return batch_u_text, batch_i_text, batch_uid, batch_iid, batch_y

'''
    函数def train_step(u_batch, i_batch, uid, iid, y_batch):
    
    功能：
        将一个batch的数据送入模型中进行计算，并根据前向传播的loss来
        进行梯度下降更新模型参数
        
    变量：
        train_op：梯度下降操作
        global_step：梯度学习率变化的全局数量  
        u_batch：该batch中的用户评论集
        i_batch：该batch中的商品评论集
        uid：该batch中全部用户的id
        iid：该batch中全部用户的id
        y_batch：该batch中用户对商品的评分
'''
def train_step(model, u_batch, i_batch, uid, iid, y_batch):
    feed_dict = {
        model.input_u: u_batch,
        model.input_i: i_batch,
        model.input_y: y_batch,
        model.input_uid: uid,
        model.input_iid: iid,
        model.dropout_keep_prob: FLAGS['dropout_keep_prob']
    }
    _, loss, diff = sess.run(
            [train_op, model.loss, model.diff],
            feed_dict 
        )

    # print(u_review_att)

    return diff

'''
    函数def test_step(u_batch, i_batch, uid, iid, y_batch):
    
    功能：
        将一个batch的数据送入模型中进行前向传播计算，不会更新模型参数
        
    变量：
        u_batch：该batch中的用户评论集
        i_batch：该batch中的商品评论集
        uid：该batch中全部用户的id
        iid：该batch中全部用户的id
        y_batch：该batch中用户对商品的评分
        batch_num：当前batch的编号。整个数据集共需要n个batch来处理，n从0开始
'''
def test_step(model, u_batch, i_batch, uid, iid, y_batch):
    feed_dict = {
        model.input_u: u_batch,
        model.input_i: i_batch,
        model.input_y: y_batch,
        model.input_uid: uid,
        model.input_iid: iid,
        model.dropout_keep_prob: 1.0
    }

    diff = sess.run(model.diff, feed_dict)

    return diff

'''
    函数save_attention(model, u_batch, i_batch, uid, iid, y_batch):
    
    功能：
        将一个batch的数据送入模型中进行前向传播计算，并返回单词级别的注意力得分word_att和
        评论级别的注意力得分review_att
        
    变量：
        u_batch：该batch中的用户评论集
        i_batch：该batch中的商品评论集
        uid：该batch中全部用户的id
        iid：该batch中全部用户的id
        y_batch：该batch中用户对商品的评分
        batch_num：当前batch的编号。整个数据集共需要n个batch来处理，n从0开始
'''
def save_attention(model, u_batch, i_batch, uid, iid, y_batch, vocabularyb):
    feed_dict = {
        model.input_u: u_batch,
        model.input_i: i_batch,
        model.input_y: y_batch,
        model.input_uid: uid,
        model.input_iid: iid,
        model.dropout_keep_prob: 1.0
    }


    u_word_att, u_review_att, i_word_att, i_review_att = sess.run([model.u_word_att, model.u_review_att, model.i_word_att, model.i_review_att], feed_dict)

    attention={}
    attention['u_id'] = uid
    attention['i_id'] = iid
    attention['u_batch'] = u_batch
    attention['i_batch'] = i_batch
    attention['y_batch'] = y_batch
    attention['vocabularyb'] = vocabularyb
    attention['u_word_att'] = u_word_att
    attention['u_review_att'] = u_review_att
    attention['i_word_att'] = i_word_att
    attention['i_review_att'] = i_review_att
    # store 'param' in hard disk
    pickle.dump(attention, open('./'+'attention.dict', 'wb'))
    print('attention writed')




'''
   def create_embedding_matrix(vocabulary, dim): 
   
   功能：
       根据指定的维度dim及词汇表（词汇表是一个dictionary，key=单词，values=单词的编号），
       从glove中读取单词编号对应的词向量来构建词向量矩阵。
   
   变量：
       embeddings_index：是一个dictionary，key=单词，values=单词对应的向量
       embedding_matrix的格式：例如编号为 i 的单词对应的词向量为embedding_matrix的第 i 行向量。
'''
def create_embedding_matrix(vocabulary, dim):
    
    print("正在构建{dim}维词向量矩阵.....".format(dim=dim))
    
    # 根据glove文件，构建词向量索引。embeddings_index是一个dictionary，key=单词，values=单词对应的向量
    embeddings_index = {}
    with open(GLOVE_DIR+ '/glove.6B.' + str(dim) + 'd.txt', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]  # 单词
            coefs = np.asarray(values[1:], dtype='float32')  # 单词对应的向量
            embeddings_index[word] = coefs  # 单词及对应的向量

    # 初始化词向量矩阵embedding_matrix
    embedding_matrix = np.random.uniform(-1.0, 1.0, (len(vocabulary)+1, dim)) # vocabulary 序列化过的单词下标最小值为1，因此词向量矩阵的长度要+1
    
    # 根据vocabulary和embeddings_index，对embedding_matrix赋值
    for word, i in vocabulary.items():
        embedding_vector = embeddings_index.get(word)  # 根据词向量字典获取该单词对应的词向量
        if embedding_vector is not None: # 如果glove有该单词的词向量，则写入词向量矩阵中
            embedding_matrix[i] = embedding_vector
    
    print("完成构建{dim}维词向量矩阵!".format(dim=dim))

    print('预训练词向量数量为：',len(embeddings_index))
    print('数据集tokend单词数为（下标从1开始）：',len(embedding_matrix))

    return embedding_matrix

'''
	def show_para(flags, param):
		功能：
			输出当前运行程序的所有参数（模型、训练、数据集参数）

		变量：
			flags：该脚本开头的全局变量
			param：数据集预处理部分生成的变量
'''
def show_para(flags, param):

    # Model Hyperparameters
    print('============ Model Hyperparameters ============')
    print('word2vec:',flags['word2vec'])
    print('word_emb_size:',flags['word_emb_size'])
    print('dropout_keep_prob:',flags['dropout_keep_prob'])
    print('init_lr:',flags['init_lr'])
    print('n_latent:',flags['n_latent'])
    print('fm_k:',flags['fm_k'])
    print('rnn_output_dim',flags['rnn_output_dim'])
    print('attention_dim',flags['attention_dim'])
    print('filter_sizes',flags['filter_sizes'])
    print('num_filters',flags['num_filters'])

    # Training parameters
    print('\n============= Training parameters =============')
    print('batch_size:',flags['batch_size'])
    print('num_epochs:',flags['num_epochs'])

    # GPU Parameters
    print('\n================ GPU Parameters ===============')
    print('allow_soft_placement:',flags['allow_soft_placement'])
    print('log_device_placement:',flags['log_device_placement'])
    print('per_process_gpu_memory_fraction:',flags['per_process_gpu_memory_fraction'])
    
    # DataSet Parameters
    print('\n============== DataSet Parameters =============')
    print('DataSet Path:',flags['para_data'])
    review_word_num = param['review_word_num']
    user_num = param['user_num']
    item_num = param['item_num']
    vocabulary_size = len(param['vocabulary'])
    review_word_num = param['review_word_num']  # the number of words per reivew
    u_review_num = param['u_review_num'] # the number of review per user
    i_review_num = param['i_review_num'] # the number of review per item
    train_size = len(param['train'])
    valid_size = len(param['valid'])
    test_size = len(param['test'])
    print('review_word_num:', review_word_num)
    print('u_review_num:', u_review_num)
    print('i_review_num:', i_review_num)
    print('user_num:', user_num)
    print('item_num:', item_num)
    print('vocabulary_size:', vocabulary_size)
    print('train_size:', train_size)
    print('valid_size:', valid_size)
    print('test_size:', test_size)
    print('')

'''
    def train(train_dataSet, batch_size):
        功能：
            根据给定的数据集train_dataSet和batch_size，对模型进行训练

        变量：
            train_dataSet：训练数据集
            batch_size：每次送入模型进行训练的样本数量
        
        返回：
            train_rmse, train_mse, train_mae
'''
def train(train_dataSet, batch_size):
    data_size_train = len(train_dataSet)
    batch_num = int(data_size_train / batch_size) # the number of total batch
    # store the difference between y and y_pre of each batch
    train_diff_list = []
    
    # 在每个epoch都对训练集data_size_train进行shuffle处理
    shuffle_indices = np.random.permutation(np.arange(data_size_train))
    shuffled_data = train_dataSet.iloc[shuffle_indices] # type(train_dataSet)= pandas datframe
    
    # 遍历 l1 个 batch
    for batch_index in range(batch_num):
        # 生成batch编号为batch_index的batch数据
        batch_u_text, batch_i_text, batch_uid, batch_iid, batch_y = \
                generate_batch(batch_index, batch_size, data_size_train, shuffled_data) 
                      
        # 计算当前batch的rmse和mae
        batch_diff = train_step(
                            model, 
                            batch_u_text, 
                            batch_i_text, 
                            batch_uid, 
                            batch_iid, 
                            batch_y
                        ) # shape=[batch,1]
        # current_step = tf.train.global_step(sess, global_step) # 学习速率第一次训练开始变化，global_steps每次自动加1
        
        # 累加各个batch的 rmse、mse、mae
        train_diff_list.extend(np.reshape(batch_diff, [-1])) # shape(batch_diff) = [？,1]

    # 计算训练集在该轮epoch的rmse、mse、mae
    train_rmse = np.sqrt(np.mean(np.square(train_diff_list)))
    train_mse = np.mean(np.square(train_diff_list))
    train_mae = np.mean(np.abs(train_diff_list))

    return train_rmse, train_mse, train_mae


'''
    def evaluate(test_dataSet, batch_size):
        功能：
            根据给定的数据集test_dataSet和batch_size，对模型进行正向传播，不训练模型

        变量：
            test_dataSet：测试数据集
            batch_size：每次送入模型进行训练的样本数量

        返回：
            test_rmse, test_mse, test_mae
'''
def evaluate(test_dataSet, batch_size):
    data_size_test = len(test_dataSet)
    batch_num = int(data_size_test / batch_size) # the number of total batch
    test_diff_list=[]
    for batch_index in range(batch_num):
        batch_u_text_test, batch_i_text_test, batch_uid_test, batch_iid_test, batch_y_test = \
                        generate_batch(batch_index, batch_size, data_size_test, test_dataSet)
            
        batch_diff = test_step(
                             model,
                             batch_u_text_test, 
                             batch_i_text_test,
                             batch_uid_test, 
                             batch_iid_test,
                             batch_y_test
                            )
        test_diff_list.extend(np.reshape(batch_diff, [-1]))
        
    # 输出测试集的rmse、mse、mae
    test_rmse = np.sqrt(np.mean(np.square(test_diff_list)))
    test_mse = np.mean(np.square(test_diff_list))
    test_mae = np.mean(np.abs(test_diff_list))

    return test_rmse, test_mse, test_mae





if __name__ == '__main__':
    
    # 输出本次实验的 全部 参数设置
    FLAGS = tf.flags.FLAGS.flag_values_dict()

    print("\nLoading data...")
    
    '''
        读入评论集数据、用户数量、商品数量、用户（商品）评论集句子截断长度、
        用户（商品）评论集的字典
    '''
    with open(FLAGS['para_data'], 'rb') as f:
        param = pickle.load(f)
        review_word_num = param['review_word_num']
        user_num = param['user_num']
        item_num = param['item_num']     
        vocabulary = param['vocabulary']
        train_dataSet= param['train']
        valid_dataSet = param['valid']
        test_dataSet= param['test']
        user_reviews=param['user_reviews'] 
        ur_iid=param['ur_iid']
        item_reviews=param['item_reviews'] 
        ir_uid=param['ir_uid']
        u_review_num=param['u_review_num'] 
        i_review_num=param['i_review_num'] 


    print('数据载入完成!\n')

    show_para(FLAGS, param)


    # 创建模型，并进行训练、测试
    with tf.Graph().as_default():
        
        # 设置session参数
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS['allow_soft_placement'],
            log_device_placement=FLAGS['log_device_placement']
        )
        # session_conf.gpu_options.allow_growth = True 
        session_conf.gpu_options.per_process_gpu_memory_fraction = FLAGS['per_process_gpu_memory_fraction']  # 设置GPU能够使用的最大显存 
        sess = tf.Session(config=session_conf)
        
        with sess.as_default():
            
            # 初始化模型: DeepCoNN
            model = DeepCoNN.build_model(
                        user_num=user_num,  # the total number of users
                        item_num=item_num,  # the total number of items
                        u_review_num = u_review_num, # the number of review per user
                        i_review_num = i_review_num, # the number of review per item
                        review_word_num = review_word_num,
                        n_latent=FLAGS['n_latent'], # embedding dimension of the user/item ID
                        vocab_size = len(vocabulary), # vocab_size
                        word_emb_size=FLAGS['word_emb_size'], # word embedding size
                        filter_sizes=list(map(int, FLAGS['filter_sizes'].split(","))), # CNN filter size
                        num_filters=FLAGS['num_filters'] # CNN filter num
                    )

            # 初始化模型: MACF
            # model = MACF.build_model(
            #             user_num=user_num,  # the total number of users
            #             item_num=item_num,  # the total number of items
            #             u_review_num = u_review_num, # the number of review per user
            #             i_review_num = i_review_num, # the number of review per item
            #             review_word_num = review_word_num,
            #             n_latent=FLAGS['n_latent'], # embedding dimension of the user/item ID
            #             fm_k=FLAGS['fm_k'], # the dimension of factorization machines
            #             vocab_size = len(vocabulary), # vocab_size
            #             word_emb_size=FLAGS['word_emb_size'], # word embedding size
            #             rnn_output_dim = FLAGS['rnn_output_dim'], # RNN output dimemsion
            #         )

            # 初始化模型: DARR
            # model = DARR.build_model(
            #             user_num=user_num,  # the total number of users
            #             item_num=item_num,  # the total number of items
            #             u_review_num = u_review_num, # the number of review per user
            #             i_review_num = i_review_num, # the number of review per item
            #             review_word_num = review_word_num,
            #             n_latent=FLAGS['n_latent'], # embedding dimension of the user/item ID
            #             fm_k=FLAGS['fm_k'], # the dimension of factorization machines
            #             vocab_size = len(vocabulary), # vocab_size
            #             word_emb_size=FLAGS['word_emb_size'], # word embedding size
            #             filter_sizes=list(map(int, FLAGS['filter_sizes'].split(","))), # CNN filter size
            #             num_filters=FLAGS['num_filters'] # CNN filter num
            #         )

            # global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS['init_lr'], beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(model.loss)
            train_op = optimizer  # .apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())
            
            # 读入预训练好的词向量到embedding矩阵
            if FLAGS['word2vec']:
                Word_Embedding_Matrix = create_embedding_matrix(vocabulary, FLAGS['word_emb_size'])
                sess.run(model.embed_matrix.assign(Word_Embedding_Matrix))
                print("tensor词向量矩阵初始化完成！\n")


            batch_size = FLAGS["batch_size"]

            # 记录开始时间
            begin_time = datetime.now()
            
            # restore the history valid result in a list
            valid_history=[]

            # restore the history test result in a list
            test_history = []

            # valid_result jitter num
            jitter_num =0

            # 重复训练 EPOCH 轮
            EPOCH = FLAGS['num_epochs']
            for epoch in range(EPOCH):

                train_rmse, train_mse, train_mae = train(train_dataSet, batch_size)
                print ("epoch {}/{}".format(str(epoch+1),EPOCH )) # epcoh下标从0开始，因此0号epoch实际为1号epoch
                print ("train:rmse,mse,mae:", train_rmse, train_mse, train_mae) 

                valid_rmse, valid_mse, valid_mae = evaluate(valid_dataSet, batch_size)
                print ("valid:rmse,mse,mae:", valid_rmse, valid_mse, valid_mae) 
                
                # early stop: if jitter_num==7, stop train and test the model with testDataset
                valid_history.append([valid_rmse, valid_mse, valid_mae])
                if len(valid_history) > 6:
                    if abs(valid_history[-2][1]-valid_history[-1][1]) < 0.01 and jitter_num < 7 :
                        test_rmse, test_mse, test_mae = evaluate(test_dataSet, batch_size)
                        print ("test:rmse,mse,mae:", test_rmse, test_mse, test_mae)
                        test_history.append([test_rmse, test_mse, test_mae])
                        jitter_num += 1 # abs(valid_mse_history[-2]-valid_mse_history[-1]) indicate it is begin to jitter
                        if jitter_num ==7:
                            break; # early stop

                # 输出运行时间（单位：秒）
                current_time = datetime.now()
                print( '运行时间(单位：秒):',(current_time - begin_time).seconds )            

            test_history = np.array(test_history)
            print()
            print("=======================    Best_Result    ===========================")
            print ("test:rmse,mse,mae:", min(test_history[:,0]), min(test_history[:,1]), min(test_history[:,2]) )
            print("=====================================================================")


            # # save the attention score，which is dedicated for 'MACF'
            # batch_u_text, batch_i_text, batch_uid, batch_iid, batch_y = \
            #     generate_batch(0, 128, len(train_dataSet), train_dataSet)

            # save_attention(model, batch_u_text, batch_i_text, batch_uid, batch_iid, batch_y, vocabulary)

            gc.collect()

    print('end')
