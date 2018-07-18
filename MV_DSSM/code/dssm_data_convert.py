# -*- coding: utf-8 -*-

import sys,time,random,operator
from numpy import array
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import tensorflow as tf
import numpy as np
import jieba
from collections import Counter
#import optparse

class DssmData(object):
    # 构造函数
    # ======================================================
    def __init__(self,source_path=None,
                 max_vocab_size=20000,batch_size=1024,
                 doc_test_source_path=None,query_test_source_path=None,pred_test_path=None):
        self.source_path = source_path
        self.min_freq=1#词表出现的次数限制
        #self.vocab_path = vocab_path
        self.pos_list = []
        ### filtering param
        self.min_word_length = 2
        self.test_rate = 0.20
        ### vocabulary info
        self.max_vocab_size = max_vocab_size#字典词汇表的最大数量
        self.vocab_size = 10000 #字典词汇表的数量
        self.vocab = {}
        ### prediction config
        self.top_k = 5
        ### 稀疏函数
        self.csr_matrix_query_train = None
        self.csr_matrix_query_test = None
        self.csr_matrix_doc_train = None
        self.csr_matrix_doc_test = None
        ## 
        self.BS = batch_size
        self.training_size = 2000
        self.testing_size = 2000
        self.training_pack_size = 2000#训练样本数量/batchsize
        self.testing_pack_size = 2000
    
    # 判断是否是汉字字符串
    # ======================================================
    def ishanzi(self,text):
        # sample: ishan(u'一') == True, ishan(u'我&&你') == False
        return all(u'\u4e00' <= char <= u'\u9fff' for char in text)

    # word-hashing策略
    # ======================================================
    def word_hashing(self,line):
       # word_hasing_list = filter(lambda x: self.ishanzi(x) or len(x) > 1, jieba.cut(line.strip().lower()))
       # word_hasing_list = filter(lambda x: self.ishanzi(x) or len(x) > 1, jieba.cut(line.strip()))     
        lines=line.split()
        sumline=[]
        trigram_list = []
        #if(len(lines)==1):
         #  return trigram_list;
        for ll in lines: 
           sumline.append(ll.replace("\x00",""))
        newline="\t".join(sumline)
        word_hasing_list = filter(lambda x: self.ishanzi(x) or len(x) > 1, jieba.cut(newline.strip()))
        for words in word_hasing_list:
            #print(words)
            if self.ishanzi(words):
                trigram_list.extend( [word for word in words] )
            else:
                trigram_words = "#%s#" % (words)
                for i in range(3,len(trigram_words)+1):
                    trigram_list.append(trigram_words[i-3:i])
            #print(trigram_list)
        return trigram_list

    # 对语料进行word-hashing，生成词典
    # ======================================================
    def gen_vocabulary_file(self):
        print('generation vocalulary dict ...')
        start = time.time()
        #with tf.Session() as sess1:
         #   self.pos_list = list(str(tf.read_file(self.source_path).eval()).split('\n'))
        with open(self.source_path,"r",encoding="utf-8",errors="ignore") as f:
            self.pos_list=f.readlines()  
        vocabulary = {}
        for line in self.pos_list:
            ## 按照word hashing策略进行切分
            tokens = self.word_hashing( line )
            for word in tokens:#出现次数,用于排序使用
                vocabulary[word] = vocabulary.get(word,0) + 1
        source_vocab = sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)
        #print(source_vocab)
        print("source vocab size: %d" % len(source_vocab)) 
        filter_vocab = filter(lambda x: len(x) == 2 and len(x[0].encode("utf8")) >= self.min_word_length and x[1] >= self.min_freq, source_vocab)
        #print("vocab path: %s" % ( self.vocab_path ))
        #tf.write_file(self.vocab_path, contents=tf.convert_to_tensor("\n".join( ["%s\t%d" % ( x[0].encode('utf8'), x[1]) for x in filter_vocab ])))
        index_vocab = [x[0] for x in filter_vocab][:self.max_vocab_size]
        vocab = dict([(x, y) for (y, x) in enumerate(index_vocab)])
        vocab_size = len(vocab)
        self.vocab_size = vocab_size
        self.vocab = vocab
        print("final vocab size: %d" % self.vocab_size)
        print("total time for vocab generation : %-3.3fs" % ( time.time() - start ))
        time.sleep(1)
    
    # 特征编码主函数
    # ======================================================
    def convert_to_vector(self):
        print('generation sparse train/test vector list ...')
        start = time.time()
        train_d_col,train_q_col,test_d_col,test_q_col = [],[],[],[]
        idx_train,idx_test = 0,0
        for idx,line in enumerate(self.pos_list):
            lines=line.split()
            sumline=[]
            if(len(lines)==1):
               continue
            for ll in lines:
               sumline.append(ll.replace("\x00",""))
            newline="\t".join(sumline)
            query_doc_list = newline.strip('\n').split("\t")
           # print(line)
           # print(query_doc_list)
           # print("=======================================")
            if len(query_doc_list) != 2:
                print("[WARN] size != 2: %s" % line.strip())
                continue
            q,d = query_doc_list
           # print("%s:%s" %(q,d))
            rate = random.random()
            vec_list_q,vec_list_d = self.word_embedding(q,rate),self.word_embedding(d,rate)
            #print(vec_list_q)
            idx_train += 1
            train_q_col.append( vec_list_q )
            train_d_col.append( vec_list_d )
            if rate <= self.test_rate:
                idx_test += 1
                test_q_col.append( vec_list_q )
                test_d_col.append( vec_list_d )
            if idx_train % 1000 == 0:
                print("processing line: %d, ..." % idx_train)
        #print(train_d_col)
       # print(train_q_col)
        ## 导出稀疏矩阵
        self.csr_matrix_query_train = self.dump_matrix(idx_train,train_q_col, "query_train")
        self.csr_matrix_doc_train = self.dump_matrix(idx_train,train_d_col, "doc_train")
        self.csr_matrix_query_test = self.dump_matrix(idx_test,test_q_col, "query_test")
        self.csr_matrix_doc_test = self.dump_matrix(idx_test,test_d_col, "doc_test")
        ## 根据训练数据规模，确定batch learning参数
        self.training_size = self.csr_matrix_doc_train.get_shape()[0]
        self.testing_size = self.csr_matrix_doc_test.get_shape()[0]
        self.training_pack_size = self.training_size //self.BS#训练样本的数量//batchsize
        self.testing_pack_size = self.testing_size // self.BS
        print("total time for sparse conversion : %-3.3fs" % ( time.time() - start )) 

    # 稀疏矩阵生成和导出
    # ======================================================
    def dump_matrix(self,idx_data,col,str_flag):
        raw_flat = [idx for idx,sublist in enumerate(col) for item in sublist]
        col_flat = [item for sublist in col for item in sublist]
        data_flat = [1 for sublist in col for item in sublist]
        print("%s length: %d" % ( str_flag,len(data_flat) ))
        matrix = coo_matrix( (array(data_flat),(array(raw_flat),array(col_flat))),
                             shape=(idx_data,self.vocab_size))
        return matrix.tocsr()

    # 把对话字符串转为向量形式
    # ======================================================
    def word_embedding(self,line,rate):
       # print(line)
       # print(rate)
        d_code = {}
        counts = Counter( self.word_hashing(line) )
        #print(self.word_hashing(line))
        for word,freq in counts.most_common():
            code = self.vocab.get(word, -1)
            #print("%s:%s:%s" %(word,freq,code))
            if code > 0: 
                d_code[code] = d_code.get(code, 0) + freq
        return [int(key) for key in d_code.keys()]

    # 校验数据维度是否满足需求
    # ======================================================
    def assert_dimension(self):
        # assert matrix dimention for sparse input training/testing data
        assert self.csr_matrix_query_train.get_shape()[1] == self.csr_matrix_query_test.get_shape()[1] == self.csr_matrix_doc_train.get_shape()[1] == self.csr_matrix_doc_test.get_shape()[1] == self.vocab_size
        assert self.csr_matrix_query_train.get_shape()[0] == self.csr_matrix_doc_train.get_shape()[0]
        assert self.csr_matrix_query_test.get_shape()[0] == self.csr_matrix_doc_test.get_shape()[0]

    # 生成tensorflow稀疏矩阵格式的batch训练数据，包括doc和query
    # ======================================================
    def pull_batch(self,size,query_data,doc_data,batch_idx):
        query_in = query_data[batch_idx * self.BS: min( size, (batch_idx + 1) * self.BS), :]
        doc_in = doc_data[batch_idx * self.BS: min( size, (batch_idx + 1) * self.BS), :]
        query_in_coo = query_in.tocoo()
        doc_in_coo = doc_in.tocoo()
        query_in_sparse = tf.SparseTensorValue(
            np.transpose([np.array(query_in_coo.row, dtype=np.int64), np.array(query_in_coo.col, dtype=np.int64)]),
            np.array(query_in_coo.data, dtype=np.float),
            np.array(query_in_coo.shape, dtype=np.int64))
        doc_in_sparse = tf.SparseTensorValue(
            np.transpose([np.array(doc_in_coo.row, dtype=np.int64), np.array(doc_in_coo.col, dtype=np.int64)]),
            np.array(doc_in_coo.data, dtype=np.float),
            np.array(doc_in_coo.shape, dtype=np.int64))
        return query_in_sparse, doc_in_sparse


    # 给tensorflow占位符生成数据并填充
    # ======================================================
    def feed_dict(self,is_train,batch_idx,dssm_data_instance):
        """Make a TensorFlow feed_dicy: maps data onto Tensor placeholders."""
        if is_train:
            query_in, doc_in = dssm_data_instance.pull_batch( dssm_data_instance.training_size,
                                            dssm_data_instance.csr_matrix_query_train,
                                            dssm_data_instance.csr_matrix_doc_train, batch_idx )
        else:
            query_in, doc_in = dssm_data_instance.pull_batch( dssm_data_instance.testing_size,
                                            dssm_data_instance.csr_matrix_query_test,
                                            dssm_data_instance.csr_matrix_doc_test, batch_idx )
        #return {dssm_model_instance.query_batch: query_in, dssm_model_instance.doc_batch: doc_in}
        return query_in, doc_in


    ##################################
    # 数据预处理主函数
    def main(self):
        start = time.time()
        self.gen_vocabulary_file()
        self.convert_to_vector()
        self.assert_dimension()
        print("total time for main data preprocession pipeline: %-3.3fs" % ( time.time() - start ) )
        #print(self.csr_matrix_query_train)


#dssmData = DssmData("D:/workspacepython/ideep_learn_proj/multiview_DSSM/part-r-00099")
#dssmData.main()
#print(dssmData.csr_matrix_query_train)


