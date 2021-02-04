#! /usr/bin/env python
# coding=gbk
from os import path
import tensorflow as tf
import numpy as np
import os,heapq,time,csv,datetime
from beaker.data_helpers import data_helpers
from beaker.text_cnn import TextCNN
from tensorflow.contrib import learn

class Predict:
    # print("\nEvaluating...\n")
    batch_size=32
    # tf.flags.DEFINE_string("checkpoint_dir","./runs/1591327187/checkpoints","Checkpoint directory from training_run")
    eval_train=False

    # Misc Parameters
    allow_soft_placement=False
    log_device_placement=False

    # Evaluation
    # ==================================================
    #
    #
    #

    def evaluat(self,checkpoint_dir,sess,x_raw,graph,categoriesArray,top_num):
        vocab_path=os.path.join(checkpoint_dir,"..","vocab")
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
            # all_scores=np.concatenate([all_scores,batch_socres])
            # tf.concat([all_predictions, batch_predictions], 0)
            # all_scores.append(batch_socres)
        # all_scores_tf=tf.convert_to_tensor(all_scores)
        # print(all_predictions)
        probs_all=sess.run(probs,{input_x:x_test,dropout_keep_prob:1.0})
        top=tf.nn.top_k(probs_all,top_num)
        indices=sess.run(top.indicess)

        s_time=time.time()
        ruslt=[]
        rusltCN=[]
        for index in indices:
            # print([idx for idx in index])
            ruslt.append([categoriesArray[idx].split('-')[1] for idx in index])
            rusltCN.append([categoriesArray[idx].split('-')[1] for idx in index])
            # print([categoriesArray[idx].split('-')[1] for idx in index])

        return ruslt,rusltCN # , tf.reduce_max(all_scores)

    def mainTest(self,sess,filecode,x_raw,graph,categoriesArray,checkpoint_filepath,top_num):
        # ruslt1, cprrect_predictionsl, Accuracy1,all_scorres1,corrunt=evaluat("./runs/1591862122/checkpoints")
        ruslt1,rusltCN=self.evaluat(checkpoint_filepath,sess,x_raw,graph,categoriesArray,top_num)
        print("{}".format(ruslt1))
        # s_time=time.time()
        # ruslt2,cprrect_predictions2, Accuracy2=evaluat("./runs/1591327828/checkpoints")
        num=0
        print("{}".format(filecode))
        for code in filecode:
            for idx in ruslt1[0]:
                # print('idx.. {}--{}'.format(idx,code))
                if int(code) == int(idx):
                    # print('num++ {}'.format(code))
                    num =num +1
        # print(': {}'.format(num))
        corrunt=num/len(filecode)
        print('单篇结果：{}'.format(corrunt))
        # e_time = time.time()
        # print('计算结果耗时：{}'.format(e_time-s_time))
        if corrunt == 1:
            return corrunt,rusltCN
        else:
            return 0,rusltCN

        # print('{}-{}'.format(cprrect_predictions2,Accuracy2))

        # def predict(self,checkpoint_path,checkpoint_filepath,testdata_dirpath):
    def predict(self,checkpoint_dir,testdata_dirpath,top_num,categoriesArray):
        # Eval Parameters

        print("\nParmeters:")
        # for attr,value in sorted(FLAGS.__flags.items()):
        #     print('{}-{}'.format(attr.upper(), value))

        # categories_path=path1
        # checkpoint_filepath=path2
        # testdata_dirpath=path3

        corruntCount=0
        testIndex=0
        # filecode = []
        checkpoint_filepath=os.path.join(checkpoint_dir,"checkpoints")
        # categories=list(open(os.path.join(checkpoint_dir,"checkpoints"),"r",encoding='utf-8').readline())
        # categoriesArray=categories[0].split(',')
        # categoriesArraysp=categoriesArray[idx].split('-')
        graph=tf.Graph()
        with graph.as_default():
            session_conf=tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement = self.log_device_placement)

            sess=tf.Session(config=session_conf)
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_filepath)
            with sess.as_default():
                saver=tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess,checkpoint_file)
            for root, dirs,files in os.walk(testdata_dirpath):
                fileCount=len(files)
                for index,file in enumerate(files):
                    file_list=[]
                    examles_positive_labels=[]
                    filecode=[]
                    filecodeCN=[]
                    fp=path.join(root,file)
                    file_examples=list(open(fp,"r",encoding='utf-8').readline())
                    file_list.extend(file_examples)
                    positive_labels=[0 for _ in range(len(categoriesArray))]
                    for categorie in categoriesArray:
                        for code in file.split('.')[0].split('-')[0].split('.'):
                            if categorie.split('-')[0]==code:
                                positive_labels[int(categorie.split('-')[1])] = int(categorie.split('-')[1])
                                filecode.append(int(categorie.split('-')[1]))
                                filecodeCN.append(int(categorie.split('-')[1]))
                    examles_positive_labels.extend([positive_labels for _ in file_examples])

                    # Split by words
                    x_raw=file_list
                    # Generate labels
                    # y_test = np.concatenate([examles_positive_labels],0)
                    # return [x_test,y]

                    # x_raw,y_test=load_data_and_labels_ex_test_2("/home/lenovo")
                    # print('y_test: {}'.format(y_test))
                    # y_test=[np.argmax(y_test,1)]
                    # y_test = [np.max(y_test, 1)]
                    # y_test = np.array(y_test)
                    # y_test = tf.cast(y_test,dtype=tf.float64)
                    # y_test = y_test.astype(float)
                    # y_test = console.log([].cancat.apply([],y_test))
                    # y_test = y_test[0]
                    # print('y_test: {}'.format(y_test))
                    corr,rusltCN=self.mainTest(self,sess,filecode,x_raw,graph,categoriesArray,checkpoint_filepath,
                                               top_num)
                    if corr == 1:
                        with open(os.path.join(checkpoint_dir, "R1ghtCede.txt"),mode="a", encoding='utf-8') as f:
                            f.write(
                                file.split(',')[0].split('-')[1]+'-'+str(filecodeCN)+"---"+str(
                                    rusltCN)+'\n\n')
                    if corr == 0:
                        with open(os.path.join(checkpoint_dir, "R1ghtCede.txt"),mode="a", encoding='utf-8') as f:
                            f.write(
                                file.split(',')[0].split('-')[1]+'-'+str(filecodeCN)+"---"+str(
                                    rusltCN)+'\n\n')
                    corruntCount =corruntCount+corr
                    testIndex=testIndex+1
                    print('测试进度：{}/{}'.format(testIndex,fileCount),
'总结果：{}'.format(corruntCount/fileCount))
