

import tensorflow as tf


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
                    vocab_size, # vocab_size
                    word_emb_size, # word embedding size
                    filter_sizes, # CNN filter size
                    num_filters # CNN filter num
                ):
        
        print()
        print("===========================    M-O-D-E-L    ===========================")
        print('                                 DARR                              ')
        print("========================================================================\n")


        self.input_u = tf.placeholder(tf.int32, [None, u_review_num, review_word_num], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, i_review_num, review_word_num], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None,1],name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        input_u = tf.reshape(self.input_u, [-1, u_review_num*review_word_num])
        input_i = tf.reshape(self.input_i, [-1, i_review_num*review_word_num])



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
            Reviews Embedding
        '''
        self.embed_matrix = tf.Variable(
                            tf.random_uniform([vocab_size+1, word_emb_size], -1.0, 1.0),
                            trainable=False,
                            name="embed_matrix",
                        ) 
        input_u_emb = tf.nn.embedding_lookup(self.embed_matrix, input_u) # shape = [None, user_review_num*u_n_words, word_emb_size]
        input_i_emb = tf.nn.embedding_lookup(self.embed_matrix, input_i) # shape = [None, item_review_num*i_n_words, word_emb_size]

        # For conv2d requires the input with shape as 4 dimension, we manually add on more dimension
        input_u_emb = tf.expand_dims(input_u_emb, -1) # shape(input_u_emb) = [None, user_review_num*u_n_words, embedding_size, 1]
        input_i_emb = tf.expand_dims(input_i_emb, -1) # shape(input_i_emb) = [None, item_review_num*u_n_words, embedding_size, 1]



        '''
            CNN for encoding the user/item information from review text
        '''
        def CNN (input_u_i_emb, filter_sizes, num_filters, word_emb_size):
            """
                使用不同大小的卷积核对 embedded_users 进行conv-maxpooling处理。

                变量：
                    pooled_outputs_u：负责存储多个不同大小的卷积核对embedded_users进行conv-maxpooling后的输出。
                    num_filters_total：卷积核总个数，等于不同大小卷积核conv-maxpooling输出的参数个数。
                    h_pool_flat_u：不同maxpooling后的输出共num_filters_total个参数，h_pool_flat_u将num_filters_total个参数展汇集成一个行向量。
            """
            pooled_outputs = []

            for i, filter_size in enumerate(filter_sizes):
                filter_shape = [filter_size, word_emb_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")         
                conv = tf.nn.conv2d(
                    input_u_i_emb,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv") # shape(conv) = [None, sequence_length - filter_size + 1, 1, num_filters]
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                word_num = input_u_i_emb.shape.as_list()[1]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, word_num - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool") # shape(pooled) = [None, 1, 1, num_filters]
                pooled_outputs.append(pooled)
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs,3) 
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # shape = [None,num_filters_total] 

            cnn_fea = tf.keras.layers.Dense(n_latent)(h_pool_flat)   # shape = [None,n_latent] 

            cnn_fea = tf.nn.dropout(cnn_fea, self.dropout_keep_prob)

            return cnn_fea 

        cnn_u = CNN(input_u_i_emb=input_u_emb, filter_sizes=filter_sizes, num_filters=num_filters, word_emb_size=word_emb_size)
        cnn_i = CNN(input_u_i_emb=input_i_emb, filter_sizes=filter_sizes, num_filters=num_filters, word_emb_size=word_emb_size)


        '''
            Attention Mechanism
        '''
        att_dim = 15 # the dimension of attention
        u_fuse = tf.add_n([cnn_u, p_u])
        i_fuse = tf.add_n([cnn_i, q_i])
        U = tf.keras.layers.Dense(att_dim,kernel_initializer='glorot_normal',  activation='relu')(u_fuse)
        I = tf.keras.layers.Dense(att_dim,kernel_initializer='glorot_normal',  activation='relu')(i_fuse)
        U_I_concat = tf.concat([U,I],1)
        att = tf.keras.layers.Dense(att_dim, kernel_initializer='glorot_normal', activation='softmax')(U_I_concat)
        UI = tf.multiply(U,I)
        F = tf.multiply(UI,att)


        '''
            Factorization Machines for Rating Prediction
        '''
        # Factorization Machines
        def FM(input_fea,fm_k):
            input_fea_dim = input_fea.shape.as_list()[1]

            WF1=tf.Variable(
                    tf.random_uniform([input_fea_dim, 1], -0.1, 0.1), name='fm1')
            Wf2=tf.Variable(
                    tf.random_uniform([input_fea_dim, fm_k], -0.1, 0.1), name='fm2')
            
            # first order feature
            one=tf.matmul(input_fea,WF1)
            
            # second order feature
            inte1=tf.matmul(input_fea,Wf2)
            inte2=tf.matmul(tf.square(input_fea),tf.square(Wf2))
            inter=(tf.square(inte1)-inte2)*0.5
            inter=tf.reduce_sum(inter,1,keepdims=True)
            
            # bias
            b=tf.Variable(tf.constant(0.1), name='bias')

            ratings =one+inter+b
            
            return ratings

        # FM predict the ratings
        y_pre = FM(input_fea=F, fm_k=6) + user_bias + item_bias + global_bias


        """
            Define the loss and difference between y_pre and self.input_y
        """
        self.diff = tf.subtract(y_pre, self.input_y)
        self.loss = tf.nn.l2_loss(self.diff)

