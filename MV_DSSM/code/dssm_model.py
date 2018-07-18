# -*- coding: utf-8 -*-  
#import cPickle as pickle
import random
import time
import sys
import numpy as np
import tensorflow as tf

class DssmModel(object):
    """
    Deep Structured Semantic Models for web search
    """
    #trigram_dim-输入的维度,输入层节点数量，字典词汇表的数量
    #epoch_steps-批量学习输入样例的数量
    #batch_size-batch的大小
    #neg_rate-负样本的数量
    #dimension-网络的维度，40*40*100
    #activation_function-激活函数
    def __init__(self,trigram_dim,epoch_steps=2,batch_size=1024, neg_rate=4,dimension=None, activation_function=None):
        ################################
        ## model parameter
        self.TRIGRAM_D = trigram_dim
        self.NEG = neg_rate                  # default 50
        self.BS = batch_size                 # default 1024
        self.dim_layer = []
        if not dimension is None:
            self.dim_layer = [int(item) for item in dimension.split("*")]
            assert len(self.dim_layer) >= 2 and len(self.dim_layer) <= 4
        self.weight_list = []  #二层、三层网络的wt
        self.bias_list = []   #二层、三层网络的bia
        self.query_act = []  #q每一层的前向传播公式，f(x*w+b)
        self.doc_act = []   #d每一层的前向传播公式，f(x*w+b)
        self.l2_reg_lambda = 0.001
        self.activation_function = activation_function
        self.epoch_steps = epoch_steps
        self.query_dim = 0.00
        self.doc_dim = 0.00
        ################################
        ## input layer
        # Shape [BS, TRIGRAM_D].  BS是batch大小，TRIGRAM_D输入维度的大小
        with tf.name_scope('input'):
            self.query_batch = tf.sparse_placeholder(tf.float32,[None,self.TRIGRAM_D], name='query_in')
            # Shape [BS, TRIGRAM_D]
            self.doc_batch = tf.sparse_placeholder(tf.float32, [None,self.TRIGRAM_D], name='doc_in')
        ################################
        ## dssm model initialization
        with tf.name_scope('model-init'):
            wh_par_range = np.sqrt(6.0 / (self.TRIGRAM_D + self.dim_layer[0]))
            self.weight_wh = tf.Variable(tf.random_uniform([self.TRIGRAM_D, self.dim_layer[0]], 
                                                           -wh_par_range, wh_par_range),name='layer-wh-weight')
            self.bias_wh = tf.Variable(tf.random_uniform([self.dim_layer[0]], 
                                                           -wh_par_range, wh_par_range),name='layer-wh-bias')
            for dim in range( 1,len(self.dim_layer) ):
                dim_par_range = np.sqrt(6.0 / (self.dim_layer[dim-1] + self.dim_layer[dim]))
                weight_dim = tf.Variable(tf.random_uniform([self.dim_layer[dim-1], self.dim_layer[dim]], 
                                                            -dim_par_range, dim_par_range),name='layer-%d-weight'%dim)
                bias_dim = tf.Variable(tf.random_uniform([self.dim_layer[dim]], 
                                                            -dim_par_range, dim_par_range),name='layer-%d-bias'%dim)
                self.weight_list.append(weight_dim)
                self.bias_list.append(bias_dim)
        ################################
        ## word-hashing layer
        with tf.name_scope('word-hashing-layer'):
            query = tf.sparse_tensor_dense_matmul(self.query_batch, self.weight_wh) + self.bias_wh
            doc = tf.sparse_tensor_dense_matmul(self.doc_batch, self.weight_wh) + self.bias_wh
            if self.activation_function is None:
                self.query_act.append( query )
                self.doc_act.append( doc )
            else:
                self.query_act.append( self.activation_function(query) )
                self.doc_act.append( self.activation_function(doc) )
        ################################
        ## multi-layer nonlinear projection
        for dim in range(1, len(self.dim_layer) ):
            self.query_dim = tf.matmul(self.query_act[-1], self.weight_list[dim-1]) + self.bias_list[dim-1]
            self.doc_dim = tf.matmul(self.doc_act[-1], self.weight_list[dim-1]) + self.bias_list[dim-1]
            if self.activation_function is None:
                self.query_act.append( self.query_dim )
                self.doc_act.append( self.doc_dim )
            else:
                if dim == len(self.dim_layer) - 1:
                    self.query_act.append( self.activation_function(self.query_dim,name='query_doc') )
                else:
                    self.query_act.append( self.activation_function(self.query_dim) )
                self.doc_act.append( self.activation_function(self.doc_dim) )
        ################################
        ## negative sampling layer
        with tf.name_scope('fd-rotate'):
            self.doc_act_fd = self.doc_act[-1]
            temp = tf.tile(self.doc_act_fd, [1, 1])#tite是将doc_act_fd复制n倍维度的，此处行复制一倍，列复制一次相当于等于原始矩阵
            for i in range(self.NEG):
                rand = int((random.random() + i) * self.BS / self.NEG)#BS是batch，random.random()返回0-1之间的实数
                self.doc_act_fd = tf.concat([self.doc_act_fd, tf.slice(temp, [rand, 0], [self.BS - rand, -1]),tf.slice(temp, [0, 0], [rand, -1])],0)


        ################################
        ## similarity layer
        with tf.name_scope('cosine-similarity-layer'):
            # Cosine similarity,tf.square(x):x向量中的每个元素均平方
            self.query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.query_act[-1]), 1, True)), [self.NEG + 1, 1])
            self.doc_norm = tf.sqrt(tf.reduce_sum(tf.square(self.doc_act_fd), 1, True))
            self.prod = tf.reduce_sum(tf.multiply(tf.tile(self.query_act[-1], [self.NEG + 1, 1]), self.doc_act_fd), 1, True)
            self.norm_prod = tf.multiply(self.query_norm, self.doc_norm)
            self.cos_sim_raw = tf.truediv(self.prod, self.norm_prod)
            self.cos_sim = tf.transpose(tf.reshape(tf.transpose(self.cos_sim_raw), [self.NEG + 1, self.BS])) * 20

        ################################
        ## loss function defination
        with tf.name_scope('loss'):
            # 转化为softmax概率矩阵。
            self.softmax_prob = tf.nn.softmax(self.cos_sim)
            # 只取第一列，即正样本列概率。
            self.hit_prob = tf.slice(self.softmax_prob, [0, 0], [-1, 1]) 
            ## l2正则
            l2_loss = self.l2_reg_lambda * ( tf.nn.l2_loss(self.weight_wh) + tf.nn.l2_loss(self.bias_wh) )
            for dim in range( 1,len(self.dim_layer) ):
                l2_loss += self.l2_reg_lambda * ( tf.nn.l2_loss(self.weight_list[dim-1]) + tf.nn.l2_loss(self.bias_list[dim-1]) )
            self.loss = -tf.reduce_sum(tf.log(self.hit_prob)) / self.BS + l2_loss
            #tf.summary.scalar('loss', self.loss)

        ################################
        ## evaluation defination: accuracy
        with tf.name_scope('accuracy'):
            self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.softmax_prob,1),0),tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)
            #tf.summary.scalar('accuracy', self.accuracy)

    ## ======================================================
    ## 打印中间数据进行可视化
    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar('sttdev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

    ## ======================================================
    ## 打印DSSM模型信息
    def debug(self):
        print ("self.query_batch:{}".format(self.query_batch))
        print ("self.query_batch shape:{},{}".format(self.query_batch.get_shape(),self.doc_batch.get_shape()))
        print ("type(self.doc_batch):",type(self.doc_batch))
        print ("weight_wh: ",self.weight_wh)
        print ("bias_wh: ",self.bias_wh)
        print ("self.doc_act_fd: ", self.doc_act_fd )
        print ("doc_y: ", self.doc_act[-1])
        print ("query_y: ", self.query_act[-1])
        print ("query_norm: ", self.query_norm)
        print ("doc_norm: ", self.doc_norm)
        print ("prod: ", self.prod)
        print ("norm_prod: ", self.norm_prod)
        print ("cos_sim: ", self.cos_sim)
        print ("softmax_prob: ", self.softmax_prob)
        print ("hit_prob: ",self.hit_prob)
        print ("loss: ",self.loss)
        print ("activation_function: ",self.activation_function)
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print (i )  # i.name if you want just a name
        """
        for var in tf.global_variables():
            print (var.name)
        """
        for var in tf.trainable_variables():
            print (var.name)
