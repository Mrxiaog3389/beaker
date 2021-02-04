#! /usr/bin/env python
# coding=gbk
from os import path
import tensorflow as tf
import numpy as np
import os,heapq,time,csv,datetime
from beaker.data_helpers import data_helpers
from beaker.text_cnn import TextCNN
from tensorflow.contrib import learn

class Predict_Real:
    batch_size=32
    eval_train=False

    # misc parameters
    allow_soft_placement=False
    log_device_placement=False

    # Evaluation
    # ==================================================
    def evaluat(self,checkpoint_filedir,sess,x_raw,graph,categoriesArray,top_num):
        vocab_path=os.path.join(checkpoint_filedir,"..","vocab")
        vocab_processor=learn.preprocessing.VocabularyProcessor(vocab_path)
        x_test=np.array(list(vocab_processor.transform(x_raw)))

        # Get the placeholders from the graph by name
        input_x=graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        socres = graph.get_operation_by_name("output/socres").outputs[0]
        probs = graph.get_operation_by_name("output/probs").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        batches=data_helpers.batch_iter(list(x_test),self.batch_size,1,shuffle=False)

        # Collect the predictions here
        all_predictions=[]
        temp_index=0

        for x_test_batch in batches:
            batch_predictions=sess.run(predictions,{input_x:x_test_batch,dropout_keep_prob:1.0})
            all_predictions=np.concatenate([all_predictions,batch_predictions])
            batch_socres=sess.run(socres,{input_x:x_test_batch,dropout_keep_prob:1.0})
            if temp_index == 0:
                temp_index=1
                all_scores=batch_socres
            else:
                all_scores=tf.concat([all_scores,batch_socres],0)
        probs_all=sess.run(probs,{input_x:x_test,dropout_keep_prob:1.0})
        top=tf.nn.top_k(probs_all,top_num)
        indices=sess.run(top.indicess)

        ruslt=[]
        rusltCN=[]
        for index in indices:
            ruslt.append([categoriesArray[idx].split('-')[1] for idx in index])
            rusltCN.append([categoriesArray[idx].split('-')[1] for idx in index])

        return ruslt,rusltCN # , tf.reduce_max(all_scores)

    def mainTest(self,sess,x_raw,graph,categoriesArray,checkpoint_filepath,top_num):
        ruslt1,rusltCN=self.evaluat(checkpoint_filepath,sess,x_raw,graph,categoriesArray,top_num)
        return rusltCN

    def predict(self,checkpoint_dir,predictcontent,top_num,categoriesArray):
        # Eval Parameters
        # categories= list(open(checkpoint_dir + "/categories.txt","r",encoding='utf-8').readline())
        # categoriesArray=categories[0].split(',')
        graph=tf.Graph()
        with graph.as_default():
            session_conf=tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement = self.log_device_placement)

            sess=tf.Session(config=session_conf)
            checkpoint_filepath=os.path.join(checkpoint_dir,'checkpoints')
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_filepath)
            with sess.as_default():
                saver=tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess,checkpoint_file)
                file_list=[]
                # split by words
                file_list.append(predictcontent)
                x_raw=file_list
                rusltCN=self.mainTest(sess,x_raw,graph,categoriesArray,checkpoint_filepath,top_num)
                return rusltCN
