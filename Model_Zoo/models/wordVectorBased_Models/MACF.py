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
                    rnn_output_dim, # RNN output dimemsion 
                ):
        self.input_u = tf.placeholder(tf.int32, [None, u_review_num, review_word_num], name="input_u") 
        self.input_i = tf.placeholder(tf.int32, [None, i_review_num, review_word_num], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None,1],name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        print()
        print("===========================    M-O-D-E-L    ===========================")
        print('                                  MACF                              ')
        print("========================================================================\n")

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
            reviews embedding
        '''
        self.embed_matrix = tf.Variable(
                            tf.random_uniform([vocab_size+1, word_emb_size], -1.0, 1.0), # 由于vocabulary 序列化过的单词下标最小值为1，因此词向量矩阵的长度要+1使得下标对齐
                            trainable=True,
                            name="embed_matrix",
                        ) 
        input_u_emb = tf.nn.embedding_lookup(self.embed_matrix, self.input_u) # shape = [None, user_review_num, u_n_words, word_emb_size]
        input_i_emb = tf.nn.embedding_lookup(self.embed_matrix, self.input_i) # shape = [None, item_review_num, i_n_words, word_emb_size]


        '''
            Process the Reviews with Bi-RNN
        '''
        def bi_rnn(rnn_type, inputs, rnn_output_dim):
            if rnn_type == 'gru':
                h = tf.keras.layers.Bidirectional(
                              tf.keras.layers.GRU(rnn_output_dim,return_sequences=True,unroll=True),
                              merge_mode='concat'
                       )(inputs) 

            elif rnn_type == 'lstm' :
                h = tf.keras.layers.Bidirectional(
                              tf.keras.layers.LSTM(rnn_output_dim,return_sequences=True,unroll=True),
                              merge_mode='concat'
                       )(inputs)
            return h # shape= (None, u_n_words, 2*rnn_output_dim) or # shape(H_d) = (None, i_n_words, 2*rnn_output_dim)


        # user review bi_rnn process
        input_u_emb = tf.reshape(input_u_emb,[-1,review_word_num, word_emb_size]) # shape = [None, u_n_words, word_emb_size]
        h_u = bi_rnn(rnn_type='gru', inputs=input_u_emb, rnn_output_dim= rnn_output_dim)# shape = [None, u_n_words, 2*rnn_output_dim]
        h_u = tf.reshape(h_u,[-1, u_review_num, review_word_num, 2*rnn_output_dim]) # # shape = [None, user_review_num, u_n_words, 2*rnn_output_dim]

        # item review bi_rnn process
        input_i_emb = tf.reshape(input_i_emb,[-1,review_word_num, word_emb_size])
        h_i = bi_rnn(rnn_type='gru', inputs=input_i_emb, rnn_output_dim= rnn_output_dim)
        h_i = tf.reshape(h_i,[-1, i_review_num, review_word_num, 2*rnn_output_dim])


        '''
            Calculate User Preference Vector U, Item Characteristic Vector I
        '''
        def build_attention_network(network_name, u_i_v, h, rnn_output_dim):
            '''
                input:
                    network_name: 'u' denotes user network, 'i' denotes item network
                    u_i_v: user embedding vector, or item embedding vector
                    h: hidden vector of RNN
                    rnn_output_dim: rnn output dimension
                return:
                    the user/item representation from the reviews
            '''
            
            # assign the local param from the global param
            words_num = review_word_num
            if network_name == 'u': 
                review_num = u_review_num
            elif network_name == 'i':
                review_num = i_review_num
               
            '''
                word-level attention
            '''   
            p_w = tf.nn.relu(tf.keras.layers.Dense(n_latent, name='p_%s_w'% network_name)(u_i_v)) # shape = (?,n_latent)
            A_w =  tf.get_variable(name="harmony_matrix_in_attention_word-level_%s" % network_name,
                                   initializer= tf.random_uniform([2*rnn_output_dim, n_latent], -1.0, 1.0),
                              )# harmony matrix in attention

            # calculate word-level attention matrix S_w
            temp = tf.matmul(p_w ,A_w,  transpose_b=True) # shape = [?,2*rnn_output_dim]
            temp1=tf.expand_dims(temp, 2) # shape = [?,2*rnn_output_dim, 1]
            h = tf.reshape(h,[-1, review_num*words_num ,2*rnn_output_dim]) # shape=[?,review_num*words_num,2*rnn_output_dim]
            temp2 = tf.matmul(h ,temp1) # shape=[?,review_num*words_num,1]
            # S_w = tf.nn.softmax(tf.nn.tanh(tf.reshape(temp2,[-1,review_num,words_num]))) # shape=[?,review_num,words_num], 注意softmax的归一化仅作用在最后一个维度所组成的向量
            # S_w = tf.nn.softmax(tf.nn.tanh(tf.matmul(h ,temp1))) # shape=[?,review_num*words_num,1]
            S_w = tf.nn.softmax(tf.reshape(temp2,[-1,review_num,words_num]))
            S_w = tf.reshape(S_w, [-1,review_num,words_num,1]) # shape = [?,review_num,words_num,1]
            S_w = tf.nn.dropout(S_w, self.dropout_keep_prob)

            # calculate word-level attention weighted sum, and get the review embedding R_u
            h = tf.reshape(h,[-1,review_num,words_num,2*rnn_output_dim]) # shape=[?,review_num,words_num,2*rnn_output_dim]
            h = tf.transpose(h,[0,1,3,2]) # shape=[?,review_num,2*rnn_output_dim,words_num]  

            R= tf.matmul(h,S_w) # shape=[?,review_num,2*rnn_output_dim,1]

            '''
                Gate mechanism
            '''
            R = tf.reshape(R,[-1,2*rnn_output_dim] )# shape=[?,2*rnn_output_dim]
            g = tf.nn.sigmoid(tf.keras.layers.Dense(2*rnn_output_dim,name='Gate_mechanism_%s' % network_name)(R))
            R_g = tf.multiply(R,g) # shape=[?,2*rnn_output_dim]
            R_g = tf.reshape(R_g,[-1,review_num,2*rnn_output_dim]) # shape=[?,review_num,2*rnn_output_dim]

            '''
                review-level attention
            '''
            p_r = tf.nn.relu(tf.keras.layers.Dense(n_latent, name='p_%s_r'% network_name)(u_i_v)) # shape = (?,n_latent)
            A_r =  tf.get_variable(name="harmony_matrix_in_attention_review-level_%s" % network_name,
                                   initializer= tf.random_uniform([2*rnn_output_dim, n_latent], -1.0, 1.0),
                              )# harmony matrix in attention

            # calculate review-level attention matrix S_r
            temp_3 = tf.matmul(p_r ,A_r,  transpose_b=True) # shape = [?,2*rnn_output_dim]
            temp_4 = tf.expand_dims(temp_3, 2) # shape = [?,2*rnn_output_dim, 1]
            temp_5 = tf.matmul(R_g ,temp_4) # shape=[?,review_num,1]
            # S_r = tf.nn.softmax(tf.nn.tanh(tf.reshape(temp_5,[-1,review_num]))) # shape=[?,review_num]
            S_r = tf.nn.softmax(tf.reshape(temp_5,[-1,review_num]))
            S_r = tf.nn.dropout(S_r, self.dropout_keep_prob)
            U_I = tf.matmul(R_g,tf.reshape(S_r,[-1,review_num,1]),transpose_a=True) # shape = [?,2*rnn_output_dim,1]
            U_I = tf.squeeze(U_I,axis=2) # shape=[?,2*rnn_output_dim]
            
            U_I = tf.keras.layers.Dense(n_latent, use_bias=False)(U_I)

            return U_I, S_w, S_r  # return the final representaion of user or item

        U, u_S_w, u_S_r = build_attention_network(network_name='u', u_i_v=p_u, h=h_u, rnn_output_dim=rnn_output_dim) # shape=[?,n_latent]
        U = tf.nn.dropout(U, self.dropout_keep_prob)
        I, i_S_w, i_S_r  = build_attention_network(network_name='i', u_i_v=q_i, h=h_i, rnn_output_dim=rnn_output_dim) # shape=[?,n_latent]
        I = tf.nn.dropout(I, self.dropout_keep_prob)

        '''
            prediction layer
        '''
        U_I_dim = U.shape.as_list()[1] # the dimension of U or I
        # Factorization Machines
        def FM(input_fea,fm_k):
            
            WF1=tf.Variable(
                    tf.random_uniform([2*U_I_dim, 1], -0.1, 0.1), name='fm1')
            Wf2=tf.Variable(
                    tf.random_uniform([2*U_I_dim, fm_k], -0.1, 0.1), name='fm2')
            
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

        '''
             gating fusion of U and I
        '''
        bias_g = tf.Variable(tf.constant(0.0, shape=[1]))
        G = tf.nn.sigmoid(tf.keras.layers.Dense(U_I_dim,use_bias=False)(U) + 
                          tf.keras.layers.Dense(U_I_dim,use_bias=False)(I) + 
                          bias_g) # shape=[?,U_I_dim]
        Z = tf.concat([tf.multiply(G,U), tf.multiply((1-G),I)],1) # shape=[?,U_I_dim]

        # Z = tf.concat([U,I], 1)

        # FM predict the ratings
        y_pre = FM(input_fea=Z, fm_k=6) + user_bias + item_bias + global_bias

        self.diff = tf.subtract(y_pre, self.input_y)
        self.loss = tf.nn.l2_loss(self.diff)


        self.u_word_att = u_S_w  # shape = [?,review_num,words_num,1]
        self.u_review_att = u_S_r# shape=[?,review_num,1]
        self.i_word_att = i_S_w  # shape = [?,review_num,words_num,1]
        self.i_review_att = i_S_r# shape=[?,review_num,1]
