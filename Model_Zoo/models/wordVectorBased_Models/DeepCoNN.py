

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
                    vocab_size, # vocab_size
                    word_emb_size, # word embedding size
                    filter_sizes, # CNN filter window size
                    num_filters # CNN filter num
                ):

        print()
        print("===========================    M-O-D-E-L    ===========================")
        print('                               DeepCoNN                              ')
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
            reviews embedding
        '''
        self.embed_matrix = tf.Variable(
                            tf.random_uniform([vocab_size+1, word_emb_size], -1.0, 1.0),
                            trainable=True,
                            name="embed_matrix",
                        ) 
        input_u_emb = tf.nn.embedding_lookup(self.embed_matrix, input_u) # shape = [None, user_review_num*u_n_words, word_emb_size]
        input_i_emb = tf.nn.embedding_lookup(self.embed_matrix, input_i) # shape = [None, item_review_num*i_n_words, word_emb_size]

        # 由于conv2d需要一个四维的输入数据，因此需要手动添加一个维度。
        input_u_emb = tf.expand_dims(input_u_emb, -1) # shape(input_u_emb) = [None, user_review_num*u_n_words, embedding_size, 1]
        input_i_emb = tf.expand_dims(input_i_emb, -1) # shape(input_i_emb) = [None, item_review_num*u_n_words, embedding_size, 1]



        '''
            Convulutional Neural Network
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
            '''
                由于shape(pooled) = [None, 1, 1, num_filters]，则每个pooled含有num_filters个数字。
                tf.concat(pooled_outputs_u,3) 表示将第3维度（下标从0开始）进行拼接，即将各个pooled
                中的num_filters个数字拼接在一起。
            '''
            h_pool = tf.concat(pooled_outputs,3) 
            '''
                由于该程序不同卷积核大小filter_sizes对应的卷积核个数num_filters都是一样的，而从
                shape(pooled) = [None, 1, 1, num_filters]知道maxpooling得到的数字个数等于num_filters，
                因此，不同卷积核大小之后的maxpooling得到的输出都是num_filters个数字。因此，不同大小的卷积核经过
                maxpooling后共有num_filters_total = num_filters * len(filter_sizes)个输出的数字。
                tf.reshape(self.h_pool_u, [-1, num_filters_total])就是将这maxpooling后共有num_filters_total个数字
                展开成一个 shape = [1,num_filters_total] 的行向量
            '''
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # shape = [None,num_filters_total] 

            cnn_fea = tf.keras.layers.Dense(n_latent)(h_pool_flat)   # shape = [None,n_latent] 

            cnn_fea = tf.nn.dropout(cnn_fea, self.dropout_keep_prob)

            return cnn_fea 

        cnn_u = CNN(input_u_i_emb=input_u_emb, filter_sizes=filter_sizes, num_filters=num_filters, word_emb_size=word_emb_size)
        cnn_i = CNN(input_u_i_emb=input_i_emb, filter_sizes=filter_sizes, num_filters=num_filters, word_emb_size=word_emb_size)


        '''
            Use FM to predict the rating
        '''
        Z = tf.concat([cnn_u, cnn_i], 1)

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
        y_pre = FM(input_fea=Z, fm_k=6) + user_bias + item_bias + global_bias


        """
            Define the loss and difference between y_pre and self.input_y
        """
        self.diff = tf.subtract(y_pre, self.input_y)
        self.loss = tf.nn.l2_loss(self.diff)


