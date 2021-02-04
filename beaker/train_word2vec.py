from gensim.models import word2vec
import os
from os import path









class word2vecclass:

    @staticmethod
    def train_word(datapath,wordmodeldir):
        if not os.path.exists(wordmodeldir):
            os.makedirs(wordmodeldir)
        sentences=word2vec.Text8Corpus(datapath)
        model=word2vec.Word2Vec(size=100,min_count=1)
        model.build_vocab(sentences)
        model.train(sentences,total_examples=model.corpus_count,epochs=1)
        model.save(os.path.join(wordmodeldir,"word2vec.model"))
        model.wv.save_word2vec_format(os.path.join(wordmodeldir,"word2vec.txt"))
        print("词向量训练完成")
