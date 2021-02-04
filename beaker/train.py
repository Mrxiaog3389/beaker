# coding=gbk
# ! /usr/bin/env python
import datetime,os
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow.contrib import learn
from beaker.data_helpers import data_helpers
from beaker.text_cnn import TextCNN

# from tensorflow.python.platform.flags import FLAGS
# from absl.flags import FLAGS

class Train:
    dev_sample_percentage=0.2
    embedding_dim=100
    filter_sizes='3.4.5'
    num_filters=128
    dropout_keep_prob=0.5
    l2_reg_lambda=0.0
    batch_size=64
    num_epochs=2
    evaluate_every=100
    checkpoint_every=100
    num_checkpoints=5
    allow_soft_placement=True
    log_device_placement=False
    categories=[]
    cheakpoint_path=''

    def __init__(self,word_model_dir):
        self.word_vec=KeyedVectors.load_word2vec_format(os.path.join(word_model_dir,"word2vec.txt"))

    def preprocess(self,data_dir,instance_path):
        #load data
        print("loadint data")
        x_text,y,categories=data_helpers.load_data_and_labels_ex_test(data_dir,instance_path)
        self.categories=categories

        # build vocabulary
        max_document_length=max([len(x.split("")) for x in x_text])
        vocab_processor=learn.preprocessing.VocabularyProcessor(max_document_length)
        x=np.array(list(vocab_processor.fit_transform(x_text)))

        # randomly shuffle data
        np.random.seed(10)
        shuffle_indices=np.random.permutation(np.arange(len(y)))
        x_shuffled=x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # split train/test set
        dev_sample_index= -1*int(self.dev_sample_percentage*float(len(y)))
        x_train,x_dev=x_shuffled[:dev_sample_index],x_shuffled[:dev_sample_index]
        y_train, y_dev = x_shuffled[:dev_sample_index], x_shuffled[:dev_sample_index]
        del x,y,x_shuffled,y_shuffled
        print('x_train: {}'.format(len(x_train)))
        print('x_dev: {}'.format(len(x_dev)))

        print('vocabulary size: {:d}'.format(len(vocab_processor.vocabulary_)))
        print('train/dev split: {:d}/{:d}'.format(len(y_train),len(y_dev)))
        return x_train,y_train,vocab_processor,x_dev,y_dev,x_text

    def train(self,x_train,y_train,vocab_processor,x_dev,y_dev,instance_path,model_name):
        #training
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement=self.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(self.word_vec.vocab),
                    # vocab_size=FLAGS.vocab_size
                    embedding_size=self.embedding_dim,
                    filter_sizes=list(map(int, self.filter_sizes.split(","))),
                    num_filters=self.num_filters,
                    l2_reg_lambda=self.l2_reg_lambda)

                # Define Training procedure 定义训练过程
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)  # 指定优化器
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Output directory for models and summaries
                # timestamp = str(int(time.time()))
                # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                out_dir = os.path.abspath(os.path.join(instance_path, "runs", model_name))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)
                # print(checkpoint_dir)
                # print(checkpoint_prefix)
                # Write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))

                # Initialize all variables   全局初始化
                sess.run(tf.global_variables_initializer())

                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: self.dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    data_all_len=0
                    data_all_acc = 0
                    data_all_loss = 0
                    dev_batches=data_helpers.batch_iter(list(zip(x_dev,y_dev)),32,self.num_epochs)
                    for dev_batch in dev_batches:
                        batch_len=len(dev_batch)
                        dev_x_batch,dev_y_batch=zip(*dev_batch)
                        feed_dict = {
                            cnn.input_x: x_batch,
                            cnn.input_y: y_batch,
                            cnn.dropout_keep_prob: 1.0
                        }
                        step, summaries, loss, accuracy = sess.run(
                            [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                            feed_dict)
                        data_all_len=data_all_len+1
                        data_all_acc=data_all_acc+accuracy
                        data_all_loss=data_all_loss+loss
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, data_all_loss / data_all_len,
                                                                    data_all_acc / data_all_len))
                    if writer:
                        writer.add_summary(summaries, step)

                # Generate batches
                print("\nGenerate batches")
                batches = data_helpers.batch_iter(
                    list(zip(x_train, y_train)), self.batch_size, self.num_epochs)
                # Training loop. For each batch...开始训练了
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % self.evaluate_every == 0:
                        print("\nEvaluation:")
                        # dev_batches=data_helpers.batch_iter(list(zip(x_dev,y_dev)),32,FLAGS.num_epochs)
                        # for dev_batch in dev_batches:
                        #     dev_x_batch, dev_y_batch = zip(*dev_batch)
                        #     dev_step(dev_x_batch, dev_y_batch,writer=dev_summary_writer)
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    if current_step % self.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        self.cheakpoint_path=path
                        print("Saved model checkpoint to {}\n".format(path))
                with open(os.path.join(out_dir,"categories.txt"),mode='w',encoding='utf-8') as f:
                    f.write(','.join([c for c in self.categories]))

    def main(self,data_dir,instance_path,word_model_dir,model_name):
        x_train, y_train, vocab_processor, x_dev, y_dev=self.preprocess(data_dir,instance_path)
        self.train(x_train, y_train, vocab_processor, x_dev, y_dev,instance_path,model_name)
