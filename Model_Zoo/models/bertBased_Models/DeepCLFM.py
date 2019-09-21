import tensorflow as tf
import os
import numpy as np


class build_model(object):
    def __init__(
                    self,
                    user_num,  # the total number of users
                    item_num,  # the total number of items
                    u_review_num , # the number of review per user
                    i_review_num , # the number of review per item
                    review_word_num ,
                    n_latent, # embedding dimension of the user/item ID
                    fm_k, # the dimension of factorization machines
                    rnn_output_dim, # RNN output dimemsion
                    attention_dim # attention dimension
                ):

        print()
        print("===========================    M-O-D-E-L    ===========================")
        print('                               DeepCLFM                              ')
        print("========================================================================\n")


        self.input_u = tf.placeholder(tf.float32, [None, u_review_num, 768], name="input_u") # 768为bert模型句子embedding的输出维度
        self.input_i = tf.placeholder(tf.float32, [None, i_review_num, 768], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None,1],name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


        '''
            global bias, user ID bias, item ID bias
        '''
        global_bias = tf.Variable(tf.constant(0.0, shape=[1]))
        user_bias_W = tf.get_variable(
                        name='user_bias_W',
                        shape=[user_num, ],
                        initializer=tf.zeros_initializer(),
                        regularizer=tf.contrib.layers.l2_regularizer(0.2)
                       )
        user_bias =  tf.nn.embedding_lookup(
                        user_bias_W,
                        self.input_uid,
                        name='user_bias'
                       )  # shape = {?,1}
        item_bias_W = tf.get_variable(
                        name='item_bias_W',
                        shape=[item_num, ],
                        initializer=tf.zeros_initializer(),
                        regularizer=tf.contrib.layers.l2_regularizer(0.2)
                       )
        item_bias =  tf.nn.embedding_lookup(
                        item_bias_W,
                        self.input_iid,
                        name='item_bias'
                       ) # shape = {?,1}

        '''
            user ID embedding, item ID embedding
        ''' 
        user_matrix_W = tf.get_variable(
                          name="user_matrix_W",
                          initializer=tf.random_uniform([user_num, n_latent], -1.0, 1.0),
                          regularizer=tf.contrib.layers.l2_regularizer(0.2)             
                         )
        p_u = tf.nn.embedding_lookup(user_matrix_W, self.input_uid)
        p_u = tf.reshape(p_u,[-1,n_latent]) # shape = (?,n_latent)

        item_matrix_W = tf.get_variable(
                          name="item_matrix_W",              
                          initializer=tf.random_uniform([item_num, n_latent], -1.0, 1.0),
                          regularizer=tf.contrib.layers.l2_regularizer(0.2),                
                         )
        q_i = tf.nn.embedding_lookup(item_matrix_W, self.input_iid)
        q_i = tf.reshape(q_i,[-1,n_latent]) # shape = (?,n_latent)


        '''
            compress the bert feature dimension from 768 to 256, then process each review embedding by bi-rnn, 
            finally summarize the reviews of a user(item) review set into a document embedding
        '''
        def bi_rnn_attention(rnn_type, inputs, n_latent):
            # shape(inputs) = [None, seq_length, dim], seq_length为每样本对应的评论数，dim为每条评论隐向量的维度（dim最初为768，是bert的输出维度）
            inputs = tf.keras.layers.Dense(512)(inputs)
            inputs = tf.keras.layers.Dense(256)(inputs) # shape(inputs) = [None, seq_length, 256]

            if rnn_type == 'gru':
                H_d = tf.keras.layers.Bidirectional(
                              tf.keras.layers.GRU(rnn_output_dim,return_sequences=True,unroll=True),
                              merge_mode='concat'
                       )(inputs) # shape(H_d) = (None, seq_length, 2*rnn_output_dim)

            elif rnn_type == 'lstm' :
                H_d = tf.keras.layers.Bidirectional(
                              tf.keras.layers.LSTM(rnn_output_dim,return_sequences=True,unroll=True),
                              merge_mode='concat'
                       )(inputs) # shape(H_d) = (None, seq_length, 2*rnn_output_dim)

            a_s = tf.nn.softmax(
                        tf.keras.layers.Dense(1, name='w_s2')(
                            tf.tanh(tf.keras.layers.Dense( attention_dim, name='w_s1')(H_d))
                        )
                    ) # shape(a_s) = (None, seq_length, 1)
            d = tf.matmul(a_s, H_d, transpose_a=True) # shape(d)= (None, 1, 2*rnn_output_dim)
            d = tf.reshape(d, [-1, 2*rnn_output_dim]) # shape(d)= (None, 2*rnn_output_dim)
            document_fea = tf.keras.layers.Dense(n_latent, name='document_fea',
                                                                 use_bias=True, 
                                                              # activation='relu'
                                                )(d) # shape(document_fea) = [None, n_latent]
            document_fea = tf.nn.dropout(document_fea, self.dropout_keep_prob)
            return document_fea
        
        text_u = bi_rnn_attention(rnn_type='gru', inputs = self.input_u, n_latent = n_latent) # shape(text_u) = [None, 1, n_latent]
        text_i = bi_rnn_attention(rnn_type='gru', inputs = self.input_i, n_latent = n_latent) # shape(text_i) = [None, 1, n_latent]


        '''
            feature fusion and predict the rating
        '''
        text_u = tf.nn.relu(text_u)
        text_i = tf.nn.relu(text_i)

        # first order feature
        u_first = tf.add_n([text_u, p_u])
        i_first = tf.add_n([text_i, q_i])

        """
            函数FM_second_fea(self, text_fea, LFM_fea):   
                计算 text_fea 和 LFM_fea FM（因子分解机）的二阶特征
            shape(text_fea) = [None, n_latent]
            shape(LFM_fea) = [None, n_latent]
            return: shape(second_fea) = [None, fm_k]
        """
        def FM_second_fea(text_fea, LFM_fea, n_latent, fm_k):            
            z=tf.nn.relu(tf.concat([text_fea, LFM_fea],1))

            Wf2=tf.Variable(
                      tf.random_uniform([n_latent*2, fm_k], -0.1, 0.1),
                      name='fm2'
                  )
            inte1=tf.matmul(z,Wf2) # shape(inte1)=[None,fm_k]
            inte2=tf.matmul(tf.square(z),tf.square(Wf2)) # shape(inte2)=[None,fm_k]
            second_fea=(tf.square(inte1)-inte2)*0.5 # shape(second_fea)=[None,fm_k]
            second_fea=tf.nn.dropout(second_fea,self.dropout_keep_prob)
            return second_fea

        # second order feature
        u_second = FM_second_fea(text_u, p_u, n_latent, fm_k)
        i_second = FM_second_fea(text_i, q_i, n_latent, fm_k)
        
        # deep feature
        U = tf.concat([u_first, u_second],1)
        I = tf.concat([i_first, i_second],1)

        # rating prediciton
        y_pre = tf.keras.layers.Dense(1, use_bias=False)(tf.multiply(U,I)) + user_bias + item_bias + global_bias


        '''
            Define the loss and difference between y_pre and self.input_y
        '''
        self.diff = tf.subtract(y_pre, self.input_y)
        self.loss = tf.nn.l2_loss(self.diff)
