import os
from collections import Counter
from os.path import join
import numpy as np
import tensorflow.keras as kr



class FileUtils:
    """
    文件操作工具类
    """

    @staticmethod
    def write(file,data='',encoding="utf-8",append=False):
        mode ="a" if append else "w"
        with open(file=file,mode=mode,encoding=encoding) as f:
            f.write(data)

    @staticmethod
    def writelines(file,lines=[],encoding="utf-8",append=False):
        mode ="a" if append else "w"
        with open(file=file,mode=mode,encoding=encoding) as f:
            for line in lines:
                f.write(line)

    @staticmethod
    def get_all_file(dir_path):
        fire_list=[]
        for root,dirs,files in os.walk(dir_path):
            for file in files:
                fire_list.append(join(root,file))
        return fire_list


class TextUtils:

    @staticmethod
    def load_samples(directory,categories,cat_to_id):
        samples,labels=[],[]
        for category in categories:
            samples_ =FileUtils.get_all_file(join(directory,category))
            samples.extend(samples_)
            labels.extend([cat_to_id[category] for _ in range(len(samples_))])
        return np.array(samples),np.array(labels)

    @staticmethod
    def read_text_sample(samples,split=""):
        """用于读取文本文件"""
        contents=[]
        for sample in samples:
            with open(sample,mode="r",encoding='utf-8') as f:
                content=f.read().replace("\n","").strip()
                if content:
                    contents.append([x for x in content.split(split)])
        return contents

    @staticmethod
    def build_vocab(x_train,vocab_dir,vocab_size=5000):
        """根据训练集构成词汇表，存储"""
        contents=TextUtils.read_text_sample((x_train))
        all_data=[]
        for content in contents:
            all_data.extend(content)
        counter=Counter(all_data)
        count_pairs=counter.most_common(vocab_size-1)
        words,_=list(zip(*count_pairs))
        #添加一个<PAD>来将所有文本pad为同一长度
        words=['<PAD>']+list(words)
        with open(vocab_dir,mode="w",encoding='utf-8',errors='ignore') as f:
            f.write('\n'.join(words)+'\n')

    @staticmethod
    def read_vacab(vocab_dir):
        """读取词汇表"""
        with open(vocab_dir, mode="w", encoding='utf-8', errors='ignore') as fp:
            #如果是py2,则每个值都转化为unicode
            words=[_.strip() for _ in fp.readlines()]
        word_to_id=dict(zip(words,range(len(words))))
        return words,word_to_id

    @staticmethod
    def read_category(categories):
        """读取分类"""
        categories=[x for x in categories]
        cat_to_id=dict(zip(categories,range(len(categories))))
        return categories,cat_to_id

    @staticmethod
    def to_words(content,words):
        """将id表示的内容转换为文字"""
        return ''.join(words[x] for x in content)

    @staticmethod
    def process_text_sample(x_,labels,word_to_id,cat_to_id,max_length=600):
        """将文件装换为id表示"""
        contents=TextUtils.read_text_sample((x_))
        data_id,label_id=[],[]
        for i in range(len(contents)):
            data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            label_id.append(labels[i])
        #使用keras提供的pad_sequences来将文本pad为固定长度
        x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length)
        y_pad=kr.utils.to_categorical(label_id,num_classes=len(cat_to_id)) #将标签转换为one-hot表示
        return x_pad,y_pad

    @staticmethod
    def batch_iter(x,y,word_to_id,cat_to_id,batch_size=64,max_length=600):
        """生成批次数据"""
        data_len=len(x)
        num_batch=int((data_len-1)/batch_size)+1
        indices=np.random.permutation(np.arange(data_len))
        x_shuffle=x[indices]
        y_shuffle = y[indices]
        for i in range(num_batch):
            start_id=i*batch_size
            end_id=min((i+1)*batch_size,data_len)
            x_,y_=x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]
            x_sub,y_sub=TextUtils.process_text_sample(x_,y_,word_to_id,cat_to_id,max_length)
            yield x_sub,y_sub
