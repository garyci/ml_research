# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os
import time
import datetime
from dssm.dssm_data_convert import DssmData
from dssm.dssm_model import DssmModel
from dssm.dssm_data_convert_predict import DssmDataPredict
from tensorflow.contrib import learn

# training param dict
# ==================================================
#激活函数集合
activation_function_dict = {
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh,
    'relu6': tf.nn.relu6,
    'crelu': tf.nn.crelu,
    'elu': tf.nn.elu
}
#优化参数设置
optimizer_dict = {
    'sgd':      lambda l_rate: tf.train.GradientDescentOptimizer(learning_rate=l_rate),
    'adadelta': lambda l_rate: tf.train.AdadeltaOptimizer(learning_rate=l_rate,rho=0.95, epsilon=1e-08),
    'adagrad':  lambda l_rate: tf.train.AdagradOptimizer(learning_rate=l_rate,initial_accumulator_value=0.1),
    'adam':     lambda l_rate: tf.train.AdamOptimizer(learning_rate=l_rate,beta1=0.9, beta2=0.999, epsilon=1e-08),
    'rmsprop':  lambda l_rate: tf.train.RMSPropOptimizer(learning_rate=l_rate,decay=0.9, momentum=0.0, epsilon=1e-10)
}

# Parameters
# ==================================================
#tf.app.flags.DEFINE_string("data_dir","","directory of data")
tf.app.flags.DEFINE_string('train_dir', 'hdfs://ns1-backup/user/shixi_wangdi5/text_cnn_train/output',
                            """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('log_dir', 'hdfs://ns1-backup/user/shixi_wangdi5/text_cnn_train/output',
                            """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('model_path', 'D:/workspacepython/ideep_learn_proj/multiview_DSSM/model',
                           """Directory where to write save model .""")
tf.app.flags.DEFINE_string('model_name', 'dssmmodel',
                           """save model name.""")
# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                            """Comma-separated list of hostname:port pairs"""
                            """you can also specify pattern like ps[1-5].example.coma""")
tf.app.flags.DEFINE_string("worker_hosts", "",
                            """Comma-separated list of hostname:port pairs,"""
                            """you can also specify worker[1-5].example.co""")
# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", """One of 'ps', 'worker'""")
tf.app.flags.DEFINE_integer("task_index", 0, """Index of task within the job""")
# Flags for defining dssm model
tf.app.flags.DEFINE_string( "positive_path","a.txt",
                            """positive data path for dssm training""")
tf.app.flags.DEFINE_string( "dimension","3*3*128",
                            """dssm network dimensions of multi-layers, format example: 400,400,120""")
tf.app.flags.DEFINE_integer("max_vocab_size", 30000,
                            """maximum word-hashing vocabulory size, default 20000""")
tf.app.flags.DEFINE_integer("batch_size", 1,
                            """input sample num for batch learning, default 1024""")
tf.app.flags.DEFINE_integer("epoch_steps", 100,
                            """input sample num for batch learning, default 1024""")
tf.app.flags.DEFINE_integer("neg_rate", 1,
                            """negative sampling rate, default 50""")
tf.app.flags.DEFINE_integer("max_steps", 20000,
                            """maximum iterator number for dssm training, default 2000""")
tf.app.flags.DEFINE_float(  "learning_rate", 0.01,
                            """learning rate for dssm training, default 0.01""")
tf.app.flags.DEFINE_string( "activation_function", 'relu',
                            """activation function for dssm model"""
                            """support list: %s"""% ( ",".join(activation_function_dict.keys())))
tf.app.flags.DEFINE_string( "optimizer", 'adadelta',
                            """optimizer for dssm model, support list: %s""" % ( ",".join(optimizer_dict.keys())) )
tf.app.flags.DEFINE_integer("summary_period", 15, "Seconds to save a summary.")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# FLAGS.flag_values_dict()

# function: check input flags
# ==================================================
def check_flag():#检验dict的有效性
    print("\nParameters:")
    # for k in FLAGS:#tensflow1.4之后使用这种方式
    #     v = FLAGS[k].value
    #
    #     print("{}={}".format(k, v))
    #
    # for attr, value in sorted(FLAGS.__flags.items()):
    #     print("{}={}".format(attr.upper(), value))

    act_func = activation_function_dict.get(FLAGS.activation_function)

    opt = optimizer_dict.get(FLAGS.optimizer)
    if act_func is None:
        print("\nactivation_function: %s is not in support list: %s\n" % (
                FLAGS.activation_function, ",".join(activation_function_dict.keys())))
    if opt is None:
        print("\noptimizer: %s is not in support list: %s\n" % (
                FLAGS.optimizer, ",".join(optimizer_dict.keys())))

# function: main dssm training pipeline
# ======================================================
def training():
    # Model and Data Instance Initialization
    print("----------------------------")
    dssm_data_instance = DssmData(FLAGS.positive_path, FLAGS.max_vocab_size, FLAGS.batch_size)
    dssm_data_instance.main()

    dssm_data_predict = DssmDataPredict()
    dssm_data_predict.vocab = dssm_data_instance.vocab
    dssm_data_predict.vocab_size = dssm_data_instance.vocab_size
    dssm_data_predict.BS=1

    print("training_pack_size: ", dssm_data_instance.training_pack_size)
    #dssm_model_instance = DssmModel(dssm_data_instance.vocab_size,FLAGS.max_steps, FLAGS.epoch_steps,
                                             # FLAGS.batch_size,   FLAGS.neg_rate,  FLAGS.dimension,
                                              #activation_function = activation_function_dict.get(FLAGS.activation_function))

    dssm_model_instance = DssmModel(dssm_data_instance.vocab_size,FLAGS.epoch_steps,FLAGS.batch_size, FLAGS.neg_rate, FLAGS.dimension,
                                    activation_function=activation_function_dict.get(FLAGS.activation_function))

    #dssm_model_instance.debug()
    # Optimizer 先实例化一个优化函数，并基于一定的学习率进行梯度优化训练
    # 设置一个用于记录全局训练步骤的单值,global_step
    optimizer = optimizer_dict.get(FLAGS.optimizer)(FLAGS.learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    grads_and_vars = optimizer.compute_gradients(dssm_model_instance.loss)
    loss_summary=dssm_model_instance.loss
    acc_summary = dssm_model_instance.accuracy
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    # loss_summary = tf.summary.scalar('loss',dssm_model_instance.loss)
    # acc_summary = tf.summary.scalar('accuracy',dssm_model_instance.accuracy)
   # train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    #初始化tensorflow持久化类
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        next_summary_time = time.time() + FLAGS.summary_period
        # Actual execution
        train_step = 0
        #while  train_step < FLAGS.max_steps:
        for train_step in range(FLAGS.max_steps):
            query_in,doc_in = dssm_data_instance.feed_dict( True,
                                                            train_step % dssm_data_instance.training_pack_size,
                                                            dssm_data_instance )
            batch_data_dict = {
                            dssm_model_instance.query_batch: query_in,
                            dssm_model_instance.doc_batch:   doc_in
                            }
            _,train_step,train_loss= sess.run([train_op, global_step,dssm_model_instance.loss], feed_dict=batch_data_dict)
            #_,train_step,train_loss,train_summary = sess.run([train_op, global_step,dssm_model_instance.loss,train_summary_op], feed_dict=batch_data_dict)
            time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            pre_eval_num = 0
            if train_step % 10 == 0:
                print("{}, step {}, loss {:g}".format(time_str, train_step, train_loss))
            # if  train_step == FLAGS.max_steps:
                # saver.save(sess, os.path.join(FLAGS.model_path, FLAGS.model_name))
                #saver.save(sess,os.path.join(FLAGS.model_path,FLAGS.model_name),global_step=global_step)
            ## 每隔epoch_steps后，进行全量数据效果评估
            #if FLAGS.task_index == 0 and train_step / dssm_model_instance.epoch_steps > pre_eval_num:
            if (train_step+1) % dssm_model_instance.epoch_steps == 0:
                ### 更新evaluation number
                pre_eval_num = train_step//dssm_model_instance.epoch_steps
                print("pre_eval_num: {}".format( pre_eval_num ))
                epoch_loss,epoch_acc,epoch_sample = 0.0,0.0,0

                for i in range(dssm_data_instance.training_pack_size):
                    query_in,doc_in = dssm_data_instance.feed_dict( True, i, dssm_data_instance )

                    dssm_data_predict.convert_to_vector("b.txt", 1,dssm_model_instance,sess)
                    return

                    batch_data_dict = {
                            dssm_model_instance.query_batch: query_in,
                            dssm_model_instance.doc_batch: doc_in
                            }

                    loss_v,correct_pred_v,softmax_prob,cos_sim = sess.run(
                                                     [dssm_model_instance.loss,
                                                      dssm_model_instance.correct_prediction,
                                                      dssm_model_instance.softmax_prob,
                                                      dssm_model_instance.cos_sim],
                                                      feed_dict=batch_data_dict)



                    epoch_loss += loss_v
                    epoch_acc += correct_pred_v.sum()
                    epoch_sample += correct_pred_v.size
                epoch_loss /= dssm_data_instance.training_pack_size
                acc = 100.0*(epoch_acc/epoch_sample)
                time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("{}, step {}, loss: {:g}, acc: {}/{}/{:.2f}%%".format(time_str,
                                    train_step,   epoch_loss, epoch_acc, epoch_sample, acc))

# ==================================================
# main dssm training pipeline
# ==================================================
if __name__ == '__main__':
    print("==============================")
    check_flag()
    training()