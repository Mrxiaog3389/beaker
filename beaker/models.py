# coding:uth-8
import datetime
import os,time
from datetime import timedelta
from os import path
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr
from sklearn import metrics
# from tensorflow.python.platform.flaggs import FLAGS
from beaker.data_helpers import data_helpers

from beaker.commons import TextUtils


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.softmax = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.softmax, 1)  # 预测类别
        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表达小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class TextRNN(object):
    """文本分类，RNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


class CnnModel:
    def __init__(self,categories,vocab_dir,model_path):
        self.config = TCNNConfig()
        self.config.num_classes = len(categories)
        self.categories, self.cat_to_id = TextUtils.read_category(categories)
        self.words, self.word_to_id = TextUtils.read_vacab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=model_path)  # 读取保存的模型

    def categories_len(self):
        return len(self.categories)

    def predict(self,message):
        content=str(message)
        data=[self.word_to_id[x] for x in content if x in self.word_to_id]
        feed_dict={
            self.model.input_x:kr.preprocessing.sequence.pad_sequences([data],self.config.seq_length),
            self.model.keep_prob:1.0
        }
        y_pred_cls=self.session.run(self.model.y_pred_cls,feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]

    def predict_top(self,message,top=3):
        content=str(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }
        probabilitys = self.session.run(self.model.softmax, feed_dict=feed_dict)
        idxs=probabilitys.argsort()[0][-top:][::-1]
        return [self.categories[idx] for idx in idxs]


class Cnntrain:

    def get_time_dif(self,start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def evaluate(self,model,sess,x_,y_,word_to_id,cat_to_id,max_length=600):
        """评估在某一数据上的准确率和损失"""
        data_len=len(x_)
        batch_eval=TextUtils.batch_iter(x_,y_,word_to_id,cat_to_id,128,max_length)
        total_loss=0.0
        total_acc = 0.0
        for x_batch,y_batch in batch_eval:
            batch_len=len(x_batch)
            feed_dict={
                model.input_x:x_batch,
                model.input_y: y_batch,
                model.keep_prob:1.0
            }
            loss,acc=sess.run([model.loss,model.acc],feed_dict=feed_dict)
            total_loss += loss+batch_len
            total_acc += acc+batch_len
        return total_loss/data_len,total_acc/data_len

    def train(self,config,model,x_train,y_train,x_val,y_val,save_path,tb_dir,word_to_id,cat_to_id):
        print('configuring tensor board and saver')
        tf.summary.scalar("loss",model.loss)
        tf.summary.scalar("accuracy", model.acc)
        merged_summary=tf.summary.merge_all()
        writer=tf.summary.FileWriter(tb_dir)
        #配置Server
        saver=tf.train.Saver
        print('loading training and validation commons')
        #载入训练集与验证集
        start_time=time.time()
        time_dif=self.get_time_dif(start_time)
        print('time usage:',time_dif)
        #创建session
        session=tf.Session()
        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph)
        print('training and evaluating...')
        start_time=time.time()
        total_batch=0   #总批次
        best_acc_val=0.0  #最佳验证集准确率
        last_improved=0  #记录上一次提升批次
        require_improvement=1000  #如果超过1000轮未提升，提前结束训练
        #run
        flag=False
        for epoch in range(config.num_epochs):
            print('epoch:',epoch+1)
            batch_train=TextUtils.batch_iter(x_train,y_train,
                                             word_to_id,cat_to_id,config.batch_size,config.seq_length)
            for x_batch,y_batch in batch_train:
                feed_dict={
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.keep_prob: config.dropout_keep_prob
                }
                if total_batch % config.save_par_batch == 0:
                    # 每多少轮次将训练结果写入tensorboard scalar
                    s=session.run(merged_summary,feed_dict=feed_dict)
                    writer.add_summary(s,total_batch)
                if total_batch % config.print_par_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    feed_dict[model.keep_prob]=1.0
                    loss_train,acc_train=session.run([model.loss,model.acc],feed_dict=feed_dict)
                    loss_val,acc_val=self.evaluate(model,session,
                                                   x_val,y_val,word_to_id,cat_to_id,config.seq_length)
                    if acc_val > best_acc_val:
                        #保存最好结果
                        best_acc_val=acc_val
                        last_improved=total_batch
                        saver.save(sess=session,save_path=save_path)
                        improved_str='*'
                    else:
                        improved_str = ''
                    time_dif=self.get_time_dif(start_time)
                    msg='iter: {0:>6},train loss: {1:>6.2},train acc:{1:>7.2%},'\
                        +'var loss；{3:>6.2}，Val Acc:{4:>7.2%},time:{5} {6}'
                    print(msg.format(total_batch,loss_train,acc_train,loss_val,acc_val,time_dif,improved_str))
                feed_dict[model.keep_prob]=config.dropout_keep_prob
                session.run(model.optim,feed_dict=feed_dict) #运行优化
                total_batch += 1
                if total_batch - last_improved > require_improvement:
                    #验证集正确率长期不提升，提前技术训练
                    print('no optimization for a long time ,auto-stopping...')
                    flag = True
                    break # 跳出循环
            if  flag:  #同上
                break

    def test(self,config,model,x_test,y_test,save_path,word_to_id,cat_to_id,categories):
        print('loading test commons')
        start_time = time.time()
        #create session
        session =tf.Session()
        session.run(tf.global_variables_initializer())
        saver=tf.train.Saver
        saver.restore(sess=session,save_path=save_path)  #读取保存的模型
        print('testing...')
        loss_test,acc_test=self.evaluate(model,session,x_test,y_test,word_to_id,cat_to_id,config.seq_length)
        msg = 'test loss: {0:>6.2},test acc: {1:>7.2%}'
        print(msg.format(loss_test,acc_test))
        # batch size
        batch_size=128
        data_len=len(x_test)
        num_batch=int((data_len-1)/batch_size)+1
        #run
        y_test_cls=y_test
        y_pred_cls=np.zeros(shape=len(x_test),dtype=np.int32)  #保存预测模型
        for i in range(num_batch): #逐批次处理
            start_id=i*batch_size
            end_id=min((i+1)*batch_size,data_len)
            x_sub,_ = TextUtils.process_text_sample(x_test[start_id:end_id],y_test[start_id:end_id],
                                                    word_to_id,cat_to_id,config.seq_length)
            feed_dict={
                model.input_x: x_sub,
                model.input_y: 1.0,
            }
            y_pred_cls[start_id:end_id]=session.run(model.y_pred_cls,feed_dict=feed_dict)
        #评估
        print('preision, recall and fl-score...')
        print(metrics.classification_report(y_test_cls,y_pred_cls,target_names=categories))
        #混淆矩阵
        print('confusion matrix...')
        cm=metrics.confusion_matrix(y_test_cls,y_pred_cls)
        print(cm)
        # time usage
        time_dif=self.get_time_dif(start_time)
        print('time usage:',time_dif)



class TextCNN(object):
    """
    A CNN for text classification. https://github.com/XqFeng-Josie/TextCNN/blob/master/TextCNN.py
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, x_test,sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        wordEmbedding=data_helpers.getWordEmbedding(x_test)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer  #1.构建中间层，单词转化成向量的形式
        """self.W = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name="W")
           self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
           self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)"""
        self.W = tf.Variable(
            tf.cast(wordEmbedding,dtype=tf.float32,name="word2vec"),
            name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)  #增加一个维度（变成4维）[None, sequence_length, embedding_size, 1]

        # Create a convolution + maxpool layer for each filter size   #2.加入卷积层、激活层和池化层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features   #将三种filtersize的output拼接并拉平，用以全连接层
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout  #3.进行dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions   #4.全连接层（output）
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.probs = tf.nn.softmax(self.scores, name="probs")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss  #4.计算损失
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy    #5.计算准确率
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                # self.top_3=heapq.nlargest(3,self.scores,name="Top3")

            '''with tf.name_scope("top"):
            self.top_3=heapq.nlargest(3,self.scores,name="Top3")'''


# 存储已有模型
models=dict()
models_cnn=dict()


def load_model(app):
    """init models"""

    model_root_dir=path.join(app.instance_path,'model')
    for model_name in os.listdir(model_root_dir):
        try:
            model_dir=path.join(app.instance_path,'model',model_name)
            vocab_dir=path.join(model_dir,'vocab.txt')
            model_path = path.join(model_dir, 'model', model_name)  #最佳验证结果保存路径
            categories_path=path.join(model_dir, 'categories.txt')
            with open(categories_path,mode='r',encoding='utf-8') as f:
                categories=f.read().split(',')
            models[model_name]=CnnModel(categories,vocab_dir,model_path)
        except:
            continue
    print(models)


def load_model_cnn(app):
    """init models"""

    model_root_dir=path.join(app.instance_path,'runs')
    for model_name in os.listdir(model_root_dir):
        try:
            model_dir=path.join(app.instance_path,'runs',model_name)
            vocab_dir=path.join(model_dir,'vocab')
            model_path = path.join(model_dir, 'checkpoints', model_name)  #最佳验证结果保存路径
            categories_path=path.join(model_dir, 'categories.txt')
            with open(categories_path,mode='r',encoding='utf-8') as f:
                categories=f.read().split(',')
            models_cnn[model_name]=CnnModel(categories,vocab_dir,model_path)
        except:
            continue
    print(models_cnn)


def get_model(name):
    """根据模型名称获取模型"""
    return models[name]


def get_model_cnn(name):
    """根据模型名称获取模型"""
    return models_cnn[name]
