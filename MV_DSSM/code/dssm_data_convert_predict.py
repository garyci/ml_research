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

#预测时使用，即预测时，会带着did uuid
#训练时，没有did 和 uuid

class DssmDataPredict(object):
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
        lines=line.split(",")
        sumline=[]
        trigram_list = []
        for i in range(0,len(lines)):
            sumline.append(lines[i].replace("\x00", ""))
        # for ll in lines:
        #    sumline.append(ll.replace("\x00",""))
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
        return trigram_list

    #load index 对语料进行word-hashing，生成词典
    #=======================================================
    def load_vocabulary_file(self,vocabulary_file):
        dict = {}
        f = open(vocabulary_file,'r')
        line = f.readline()
        while line:
            line = line.rstrip("\n")
            lines = str(line).split("\t")
            if lines != None and len(lines)>= 2:
                word = lines[0]
                idx = lines[1]
                dict[word] =idx
            line = f.readline()

        # print("--dict ---",str(dict))
        self.vocab_size = len(dict)
        self.vocab = dict
        return dict

    # 特征编码主函数
    # ======================================================
    def convert_to_vector(self,predict_file,batch_num,dssm_model,sess):
        print('generation sparse train/test vector list ...')
        start = time.time()
        # train_d_col,train_q_col,test_d_col,test_q_col = [],[],[],[]
        did_uuid_list,did_vec_list,uuid_vec_list = [],[] ,[]
        #not cache instead of reading file readline for each
        f=open(predict_file,'r')
        line = f.readline()
        while line:
            lines=line.split()
            line = f.readline()
            sumline=[]
            if(len(lines)==1):
               continue
            for ll in lines:
               sumline.append(ll.replace("\x00",""))
            newline="\t".join(sumline)
            query_doc_list = newline.strip('\n').split("\t")
            if len(query_doc_list) != 4:
                print("[WARN] size != 4: %s" % line.strip())
                continue
            did,uuid,q,d = query_doc_list
            did_uuid_list.append(str(did)+"\t"+str(uuid))
            if(len(did_uuid_list) > batch_num):
                self.save_result_did_uuid('did_click_uuid.txt',did_uuid_list)
                did_uuid_list.clear()

            rate = random.random()
            # vec_list_q,vec_list_d = self.word_embedding(q,rate),self.word_embedding(d,rate)
            vec_list_q,vec_list_d,query_idx,doc_idx = self.query_doc_to_vec_list(q,d,rate,",")

            self.csr_matrix_query_train=self.dump_matrix(query_idx, vec_list_q, "query_train")
            self.csr_matrix_doc_train = self.dump_matrix(doc_idx, vec_list_d, "doc_train")
            self.csr_matrix_query_test = self.dump_matrix(query_idx,vec_list_q, "query_test")
            self.csr_matrix_doc_test = self.dump_matrix(doc_idx,vec_list_d, "doc_test")

            print("===1===",str(self.csr_matrix_query_train))
            query_in,doc_in=self.pull_batch(1,self.csr_matrix_query_train,self.csr_matrix_doc_train,0)
            print("===2===",str(query_in))
            batch_data_dict = {
                dssm_model.query_batch: query_in,
                dssm_model.doc_batch: doc_in
            }
            loss_v, correct_pred_v, softmax_prob, cos_sim,query_dim,doc_dim = sess.run(
                [dssm_model.loss,
                 dssm_model.correct_prediction,
                 dssm_model.softmax_prob,
                 dssm_model.cos_sim,
                 dssm_model.query_dim,
                 dssm_model.doc_dim],
                feed_dict=batch_data_dict)

            result_vec_query,result_vec_uuid = query_dim,doc_dim
            did_vec_list.append(str(did) + "\t" + str(result_vec_query))
            if len(did_vec_list) > batch_num:
                self.save_result_did_uuid('did_vec.txt', did_vec_list)
                did_vec_list.clear()

            uuid_vec_list.append(str(uuid) + "\t" + str(result_vec_uuid))
            if len(result_vec_uuid) > batch_num:
                self.save_result_did_uuid('uuid_vec.txt', uuid_vec_list)
                uuid_vec_list.clear()

        #结果保存
        self.save_result_did_uuid('did_click_uuid.txt', did_uuid_list)
        self.save_result_did_uuid('did_vec.txt', did_vec_list)
        self.save_result_did_uuid('uuid_vec.txt', uuid_vec_list)

        print("total time for sparse conversion : %-3.3fs" % ( time.time() - start ))



    #根据传入的query 和 doc返回相应的向量
    #======================================================
    def query_doc_to_vec_list(self,query,doc,rate,separator):
        query_vec_list = []
        doc_vec_list = []
        query_arr = query.split(separator)
        doc_arr = doc.split(separator)
        query_idx=0
        doc_idx=0

        for q in query_arr:
            vec_q= self.word_embedding(q, rate)
            if(len(vec_q) > 0):
                query_vec_list.append(vec_q)
                query_idx+=1

        for d in doc_arr:
            vec_d = self.word_embedding(d, rate)
            if len(vec_d) >0:
                doc_vec_list.append(vec_d)
                doc_idx+=1

        return query_vec_list,doc_vec_list,query_idx,doc_idx

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
        d_code = {}
        counts = Counter( self.word_hashing(line) )
        # print(self.word_hashing(line))
        for word,freq in counts.most_common():
            code = self.vocab.get(word, -1)
            #print("%s:%s:%s" %(word,freq,code))
            if int(code) > 0:
                d_code[code] = d_code.get(code, 0) + freq
        return [int(key) for key in d_code.keys()]

    #存储
    def save_result_did_uuid(self,sava_path,data_list):
        f = open(sava_path,"a+")
        for did_uuid in data_list:
            f.writelines(did_uuid)
            f.writelines("\n")
        f.close()

    #加载model，进行预测，并返回user的向量，和doc的向量
    def load_model_predict(self,model_path,csr_matrix_query_train,csr_matrix_doc_train):
        query_vec = []
        doc_vec = []

        return query_vec,doc_vec

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
        self.load_vocabulary_file('word2idx.txt')
        # self.convert_to_vector("b.txt",2)
        self.assert_dimension()
        print("total time for main data preprocession pipeline: %-3.3fs" % ( time.time() - start ) )
        #print(self.csr_matrix_query_train)


if __name__ == "__main__":
    dssmData = DssmDataPredict()
    dssmData.main()

