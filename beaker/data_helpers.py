import os,re,gensim
from os import path
import numpy as np

class data_helpers:
    def clean_str(self,string):
        """
        Takenization/string cleaning for all datasets excrpt for SST
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py   https://github.com/XqFeng-Josie/TextCNN/blob/master/data_helper.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def load_data_and_labels(self,positive_data_file,negative_data_file):
        """
        Loads MR polarity data from files, splits the data into words and generate labels.
        Returns split sentences and labels.
        """
        # Load data from files
        positive_examples=list(open(positive_data_file,"r",encoding='uth-8').readline())
        positive_examples=[s.strip() for s in positive_examples]
        negative_examples=list(open(negative_data_file,"r",encoding='uth-8').readline())
        negative_examples=[s.strip() for s in negative_examples]
        # Split by words
        x_text=positive_examples+negative_examples
        x_text=[self.clean_str(sent) for sent in x_text]
        # Generate labels
        positive_labels=[[0,1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y=np.concatenate([positive_labels,negative_labels],0)
        return [x_text,y]

    @staticmethod
    def load_data_and_labels_ex_test(data_dir,instance_path):
        """
        Loads MR polarity data from files, splits the data into words and generate labels.
        Returns split sentences and labels.
        """
        # Load data from files
        samples=[]
        labels=[]
        categories=[]
        print(data_dir)
        for root,dirs,files in os.walk(data_dir):
            print(files)
            for index,file in enumerate(files):
                fp=path.join(root,file)
                sample_list=list(open(fp,"r",encoding='utf-8').readline())
                # for file_example in  file_examples:
                # file_list.extend(file_example.split("\t")[1])
                samples.extend((sample_list))
                label=[0 for _ in range(len(files))]
                label[index]=1
                labels.extend([label for _ in sample_list])
                categories.append(file.split('.')[0]+'_'+str(index))
        # with open(os.path.join(categoriesdir,'categories.txt'),mode='w',encoding='utf-8') as f
        #      f.write(','.join([c for c in categories]))

        # Split by words
        x=samples
        # Generate labels
        y = np.concatenate([labels], 0)
        return x,y,categories

    @staticmethod
    def load_data_and_labels_ex_test_1(Test_data_file):
        """
        Loads MR polarity data from files, splits the data into words and generate labels.
        Returns split sentences and labels.
        """
        # Load data from files
        Test_examples=list(open(Test_data_file,"r",encoding='utf-8').readline())
        # Split by words
        x_text=Test_examples
        # Generate labels
        y=[[1] for _ in Test_examples]
        return x_text

    @staticmethod
    def load_data_and_labels_ex_test_2(TY_data_file_dir):
        """
        Loads MR polarity data from files, splits the data into words and generate labels.
        Returns split sentences and labels.
        """
        # Load data from files
        file_list=[]
        examples_positive_labels=[]
        # categories=list(open("./data/categories.txt","r",encoding='utf-8').readline())
        # categoriesArray=categories[0].split(',')
        for root,dirs,files in os.walk(TY_data_file_dir):
            # print(files)
            for index,file in enumerate(files):
                fp=path.join(root,file)
                file_examples=list(open(fp,"r",encoding='utf-8').readline())
                #print(file_examples)
                file_list.extend(file_examples)
                # positive_labels=[0 for in range(len(files))]
                """for categories in categoriesArray:
                    if categorie.split('-')[0] == file.split('.')[0]:
                    positive_labels[index] = int(categorie.split('-')[1])
                    break
                #positive_labels[index]=1
                #print(positive_labels)
                examples_positive_labels.extend([positive_labels for _ file_examples])"""

        # Split by words
        x_text=file_list
        # Generate labels
        y = np.concatenate([examples_positive_labels], 0)
        return [x_text,y]

    @staticmethod
    def getWordEmbedding(x_test,wordmodeldir):
        wordVec=gensim.models.KeyedVectors.load_word2vec_format(os.path.join(wordmodeldir,"word2vec.txt"))
        vocab=[]
        wordEmbedding=[]
        wordEmbedding.append(np.zeros(100))
        for word in x_test:
            try:
                wordarr=[]
                wordarr=word.split("")
                for word2 in wordarr:
                    word2=word2.strip()
                    vector=wordVec.wv[word2]
                    vocab.append(word2)
                    wordEmbedding.append(vector)
            except:
                print(word2+"bucunzai")
        return np.array(wordEmbedding)

    @staticmethod
    def batch_iter(data,batch_size,num_epochs,shuffle=True):  #生成batch数据
        """
        Generate a batch iterator for a dataset
        """
        data=np.array(data)
        data_size=len(data)
        num_batches_per_epoch=int(len(data)-1) /batch_size+1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices=np.random.permutation(np.arange(data_size))
                shuffle_data=data[shuffle_indices]
            else:
                shuffle_data=data
            for batch_num in range(num_batches_per_epoch):
                start_index=batch_num+batch_size
                end_index=min((batch_num+1)*batch_size,data_size)
                yield shuffle_data[start_index:end_index]
