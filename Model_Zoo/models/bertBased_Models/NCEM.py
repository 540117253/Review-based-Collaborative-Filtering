import tensorflow as tf



class build_model(object):
    def __init__(
                    self,
                    user_num,  # the total number of users
                    item_num,  # the total number of items
                    u_review_num , # the number of review per user
                    i_review_num , # the number of review per item
                    review_word_num , # the number of words per reivew
                    n_latent, # embedding dimension of the user/item ID
                    fm_k, # the dimension of factorization machines
                    attention_dim # attention dimension
                ):

        print()
        print("===========================    M-O-D-E-L    ===========================")
        print('                                 NCEM                              ')
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
            module 1: attention mechanism
        '''
        input_u = tf.keras.layers.Dense(256)(self.input_u)
        input_u = tf.keras.layers.Dense(n_latent)(input_u) # shape(inputs) = [None, seq_length, n_latent]
        a_u = tf.nn.softmax(
                    tf.keras.layers.Dense(1, name='w_1_u')(
                        tf.tanh(tf.keras.layers.Dense( attention_dim, name='w_2_u')(self.input_u))
                    )
                ) # shape(a_s) = (None, seq_length, 1)
        text_u = tf.matmul(a_u, input_u, transpose_a=True) # shape(d)= (None, 1, n_latent)
        text_u = tf.reshape(text_u, [-1, n_latent]) # shape(d)= (None, n_latent)


        input_i = tf.keras.layers.Dense(256)(self.input_i)
        input_i = tf.keras.layers.Dense(n_latent)(input_i) # shape(inputs) = [None, seq_length, n_latent]
        a_i = tf.nn.softmax(
                    tf.keras.layers.Dense(1, name='w_1_i')(
                        tf.tanh(tf.keras.layers.Dense( attention_dim, name='w_2_i')(self.input_i))
                    )
                ) # shape(a_s) = (None, seq_length, 1)
        text_i = tf.matmul(a_i, input_i, transpose_a=True) # shape(d)= (None, 1, n_latent)
        text_i = tf.reshape(text_i, [-1, n_latent]) # shape(d)= (None, n_latent)


        '''
            module 2: projection layer
        ''' 
        prob_u_text = tf.nn.softmax(tf.matmul(text_u,tf.transpose(user_matrix_W)), name = 'prob_user')
        prob_i_text = tf.nn.softmax(tf.matmul(text_i,tf.transpose(item_matrix_W)), name = 'prob_item')

        loss_1 = tf.losses.sparse_softmax_cross_entropy(labels = self.input_uid, logits = prob_u_text)
        loss_2 = tf.losses.sparse_softmax_cross_entropy(labels = self.input_iid, logits = prob_i_text)


        '''
            module 3: prediction rating by neural FM layer
        '''
        def neural_FM( fea, n_latent, fm_k):
            # first fea
            first_fea = tf.keras.layers.Dense(fm_k)(fea)
            
            # second fea
            Wf2=tf.Variable(
                      tf.random_uniform([n_latent, fm_k], -0.1, 0.1),
                      name='fm2'
                  )
            inte1=tf.matmul(fea,Wf2) # shape(inte1)=[None,fm_k]
            inte2=tf.matmul(tf.square(fea),tf.square(Wf2)) # shape(inte2)=[None,fm_k]
            second_fea=(tf.square(inte1)-inte2)*0.5 # shape(second_fea)=[None,fm_k]

            return tf.concat([first_fea,second_fea],1)

        z0 = tf.concat([p_u,q_i],1)
        z1 = tf.nn.relu(neural_FM(z0, n_latent=z0.shape[1].value, fm_k=6)) # neural_FM layer 1
        z2 = tf.nn.relu(neural_FM(z1, n_latent=z1.shape[1].value, fm_k=6)) # neural_FM layer 2
        y_pre = tf.keras.layers.Dense(1, use_bias=False, name='output_layer')(z2) +global_bias+ user_bias + item_bias

        '''
            Define the loss and difference between y_pre and self.input_y
        '''
        self.diff = tf.subtract(y_pre, self.input_y)
        self.loss = tf.nn.l2_loss(self.diff) + 0.5*loss_1 + 0.5*loss_2 


# # the number of review per user
# user_review_num=5
# # the number of review per item
# item_review_num=15
# # the number of words per reivew
# max_seq_length = 200
# # embedding dim
# n_latent = 15 

# input_u = tf.placeholder(tf.float32, [None, user_review_num, n_latent], name="input_u") # n_latent denotes the dimension of per review
# input_i = tf.placeholder(tf.float32, [None, item_review_num, n_latent], name="input_i")
# input_y = tf.placeholder(tf.float32, [None,1],name="input_y")
# input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
# input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
# dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# # input_u = tf.reshape(input_u, [-1,768]) # shape = {?,768}
# # input_i = tf.reshape(input_i, [-1,768]) # shape = {?,768}


# # the total number of users and items
# user_num = 1000
# item_num = 2000

# # global bias, user bias, item bias
# global_biase = tf.Variable(tf.constant(0.0, shape=[1]))
# user_bias_W = tf.get_variable(
#                             name='user_bias_W',
#                             shape=[user_num, ],
#                             # initializer=tf.contrib.layers.xavier_initializer()
#                             initializer=tf.zeros_initializer(),
#                             regularizer=tf.contrib.layers.l2_regularizer(0.2)
#                             )
# user_bias =  tf.nn.embedding_lookup(
#                                    user_bias_W,
#                                    input_uid,
#                                    name='user_bias'
#                                    )  # shape = {?,1}
# item_bias_W = tf.get_variable(
#         name='item_bias_W',
#         shape=[item_num, ],
#         # initializer=tf.contrib.layers.xavier_initializer()
#         initializer=tf.zeros_initializer(),
#         regularizer=tf.contrib.layers.l2_regularizer(0.2)
#         )
# item_bias =  tf.nn.embedding_lookup(
#                                    item_bias_W,
#                                    input_iid,
#                                    name='item_bias'
#                                    ) # shape = {?,1}


# # user embedding, item embedding

# user_matrix_W = tf.get_variable(
#                  # tf.random_uniform([user_num, n_latent], -1.0, 1.0),
#                  shape=[user_num, n_latent],
#                  initializer=tf.zeros_initializer(),
#                  regularizer=tf.contrib.layers.l2_regularizer(0.2),
#                  name="user_matrix_W"
#        )
# p_u = tf.nn.embedding_lookup(user_matrix_W, input_uid)
# p_u = tf.reshape(p_u,[-1,n_latent]) # shape = (?,n_latent)

# item_matrix_W = tf.get_variable(
#     # tf.random_uniform([item_num, n_latent], -1.0, 1.0),
#     shape=[item_num, n_latent],
#     initializer=tf.zeros_initializer(),
#     regularizer=tf.contrib.layers.l2_regularizer(0.2),
#     name="item_matrix_W")
# q_i = tf.nn.embedding_lookup(item_matrix_W, input_iid)
# q_i = tf.reshape(q_i,[-1,n_latent]) # shape = (?,n_latent)



# ### attention
# attention_dim = 200

# a_u = tf.nn.softmax(
#             tf.keras.layers.Dense(1, name='w_1_u')(
#                 tf.tanh(tf.keras.layers.Dense( attention_dim, name='w_2_u')(input_u))
#             )
#         ) # shape(a_s) = (None, seq_length, 1)
# text_u = tf.matmul(a_u, input_u, transpose_a=True) # shape(d)= (None, 1, n_latent)
# text_u = tf.reshape(text_u, [-1, n_latent]) # shape(d)= (None, n_latent)

# a_i = tf.nn.softmax(
#             tf.keras.layers.Dense(1, name='w_1_i')(
#                 tf.tanh(tf.keras.layers.Dense( attention_dim, name='w_2_i')(input_i))
#             )
#         ) # shape(a_s) = (None, seq_length, 1)
# text_i = tf.matmul(a_i, input_i, transpose_a=True) # shape(d)= (None, 1, n_latent)
# text_i = tf.reshape(text_i, [-1, n_latent]) # shape(d)= (None, n_latent)


# ### module 2
# prob_u_text = tf.nn.softmax(tf.matmul(text_u,tf.transpose(user_matrix_W)), name='prob_user')
# prob_i_text = tf.nn.softmax(tf.matmul(text_i,tf.transpose(item_matrix_W)), name='prob_item')

# loss_1 = tf.losses.sparse_softmax_cross_entropy(labels=input_uid, logits=prob_u_text)
# loss_2 = tf.losses.sparse_softmax_cross_entropy(labels=input_iid, logits=prob_i_text)


# ### neural FM layer
# fm_k = 6 # FM dim

# def neural_FM( fea, n_latent, fm_k):
#     # first fea
#     first_fea = tf.keras.layers.Dense(fm_k)(fea)
    
#     # second fea
#     Wf2=tf.Variable(
#               tf.random_uniform([n_latent, fm_k], -0.1, 0.1),
#               name='fm2'
#           )
#     inte1=tf.matmul(fea,Wf2) # shape(inte1)=[None,fm_k]
#     inte2=tf.matmul(tf.square(fea),tf.square(Wf2)) # shape(inte2)=[None,fm_k]
#     second_fea=(tf.square(inte1)-inte2)*0.5 # shape(second_fea)=[None,fm_k]

#     return tf.concat([first_fea,second_fea],1)

# # module 3
# z0 = tf.concat([p_u,q_i],1)
# z1 = tf.nn.relu(neural_FM(z0, n_latent=z0.shape[1].value, fm_k=6)) # neural_FM layer 1
# z2 = tf.nn.relu(neural_FM(z1, n_latent=z1.shape[1].value, fm_k=6)) # neural_FM layer 2
# pre_y = tf.keras.layers.Dense(1, use_bias=False, name='output_layer')(z2) +global_biase+ user_bias + item_bias

# loss_rating = tf.nn.l2_loss(tf.subtract(pre_y, input_y), name='loss_rating')
# loss = 0.5*loss_1 + 0.5*loss_2 + loss_rating