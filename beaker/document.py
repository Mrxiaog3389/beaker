import json
import os
from concurrent.futures import ThreadPoolExecutor
from os import path
from flask import Blueprint,request,jsonify,current_app
from sklearn.model_selection import train_test_split
from beaker.commons import TextUtils,FileUtils
from beaker.models import TCNNConfig,TextCNN,Cnntrain,get_model,load_model,load_model_cnn
from beaker.pojo import Response
from beaker.predict import Predict
from beaker.predict_real import Predict_Real
from beaker.text_cnn import TextCNN
from beaker.train import Train
from beaker.train_word2vec import word2vecclass

bp=Blueprint('document',__name__,url_prefix='/document')
executor = ThreadPoolExecutor(3)

@bp.route('/sample',methods=['POST'])
def sample():
    """
    接收请求端数据，并按一个时间为一个文件进行组织存储
    ;parm请求参数格式
        {
            "name":"sample",
            "data":{
                "id":1346268613836800,
                "name":"",
                "label":"0",
                "content":"",
                "date":1585014916672,
                "directory":"text"
            }
        }
    :return:
    """
    try:
        param=json.loads(request.get_data())
        if param['name'] != 'sample':
            return jsonify(Response(code=1,msg='service interface name error').__dict__)
        data=param['data']
        dir_path=path.join(current_app.instance_path,'dataset',data['directory'],data['label'])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # write file
        file = path.join(dir_path,data['name'])
        FileUtils.write(file=file,data='{}\n'.format(data['content']),append=True)
        return jsonify(Response(data=dir_path).__dict__)
    except:
        return jsonify(Response(code=-1, msg='service call error').__dict__)


def task_train(instance_path,name,categories,data_directory):
    print('start train...')
    # load data
    directory=path.join(instance_path,'dataset',data_directory)
    categories,cat_to_id=TextUtils.read_category(categories)
    samples,labels=TextUtils.load_samples(directory,categories,cat_to_id)
    x_train,x_test,y_train,y_test=train_test_split(samples,labels,test_size=0.2,random_state=10)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=10)
    # path
    model_dir=os.path.join(instance_path,'models',name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    vocab_dir=os.path.join(model_dir,'vocab.txt')
    save_path=os.path.join(model_dir,'model',name)  # 最佳验证结果保存路径
    tb_dir = os.path.join(model_dir, 'tensorboard') # tensor board
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    # save categories
    categories_path = os.path.join(model_dir, 'categories.txt')
    with open(categories_path,mode='w',encoding='utf-8') as f:
        f.write(','.join(categories))
    # config model
    print('configuring cnn model...')
    config=TCNNConfig()
    config.num_classes=len(categories)
    if not os.path.exists(vocab_dir):  #如果不存在词汇表,重建
        TextUtils.build_vocab(x_train,vocab_dir,config.vocab_size)
    words,word_to_id=TextUtils.read_vacab(vocab_dir)
    config.vocab_size=len(words)
    cnn=TextCNN(config)
    # train and test
    cnn_train=Cnntrain()
    cnn_train.train(config,cnn,x_train, y_train,x_val,y_val,save_path,tb_dir,word_to_id,cat_to_id)
    cnn_train.test(config, cnn, x_test, y_test, save_path, word_to_id, cat_to_id,categories)
    load_model(current_app)
    load_model_cnn(current_app)
    print('model train complete!')


@bp.route('/train',methods=['POST'])
def train():
    """
    ;parm 请求参数格式
        {
            "name":"train",
            "data":{
                "name":"textcnn-one",
                "categories":"0.1",
                "directory":"text"
            }
        }
    :return:
    """
    try:
        param = json.loads(request.get_data())
        if param['name'] != 'train':
            return jsonify(Response(code=1, msg='service interface name error').__dict__)
        data = param['data']
        task_train(current_app.instance_path,data['name'],data['categories'].split(','),data['directory'])
        return jsonify(Response(data={'msg':'train complete!'}).__dict__)
        # executor.submit(task_train,current_app.instance_path,data['name'],data['categories'].split(','))
        # return jsonify(Response(data={'msg': 'start train!'}).__dict__)
    except Exception as e:
        current_app.logger.errot(e)
        return jsonify(Response(code=-1, msg='service call error').__dict__)


@bp.route('/model',methods=['GET'])
def model():
    """
    获取已有的全部模型
    :return:
    """
    return {"id":1,"name":"model"}


@bp.route('/predict',methods=['POST'])
def predict():
    """
    参数格式
        {
            "name":"predict",
            "data":{
                "name":"textcnn",
                "data":"nichola nicklebi celabr human spirit unrel dickensian decenc turn me horror scroog",
            }
        }
    :return:
    """
    try:
        param = json.loads(request.get_data())
        if param['name'] != 'predict':
            return jsonify(Response(code=1, msg='service interface name error').__dict__)
        data = param['data']
        model=get_model(data['name'])
        return jsonify(Response(data=model.predict(data['data'])).__dict__)
    except Exception as e:
        current_app.logger.errot(e)
        return jsonify(Response(code=-1, msg='service call error').__dict__)


@bp.route('/predict/top',methods=['POST'])
def predict_top():
    """
    参数格式
        {
            "name":"predict",
            "data":{
                "name":"textcnn",
                "top":3,
                "data":"nichola nicklebi celabr human spirit unrel dickensian decenc turn me horror scroog",
            }
        }
    :return:
    """
    try:
        param = json.loads(request.get_data())
        if param['name'] != 'predict':
            return jsonify(Response(code=1, msg='service interface name error').__dict__)
        data = param['data']
        model = get_model(data['name'])
        if model.categories_len() < int(data['top']):
            msg='the number of model classifications is {}'.format(model.categories_len())
            return jsonify(Response(code=-1,data=msg,msg='error').__dict__)
        else:
            return jsonify(Response(data=model.predict_top(data['data'],int(data['top']))).__dict__)
    except Exception as e:
        current_app.logger.errot(e)
        return jsonify(Response(code=-1, msg='service call error').__dict__)


@bp.route('/word/train',methods=['POST'])
def word_train():
    """
    ;parm 请求参数格式
        {
            "name":"wordtrain",
            "data":{
                "directory":"D:/data/cutdata.txt"
            }
        }
    :return:
    """
    try:
        param = json.loads(request.get_data())
        if param['name'] != 'wordtrain':
            return jsonify(Response(code=1, msg='service interface name error').__dict__)
        data = param['data']
        data_path=data['directory']
        word_model_dir=os.path.join(current_app.instance_path,'word2vec')
        word2vecclass.train_word(data_path,word_model_dir)
        return jsonify(Response(msg='word train complete').__dict__)
    except Exception as e:
        current_app.logger.errot(e)
        return jsonify(Response(code=-1, msg='service call error').__dict__)


@bp.route('/cnn/train',methods=['POST'])
def train_cnn():
    """
    ;parm 请求参数格式
        {
            "name":"traincnn",
            "data":{
                "name":"textcnn"
                "directory":"D:/textcnn/dataset/train"
            }
        }
    :return:
    """
    try:
        param = json.loads(request.get_data())
        if param['name'] != 'traincnn':
            return jsonify(Response(code=1, msg='service interface name error').__dict__)
        data = param['data']
        model_name = data['name']
        word_model_dir = os.path.join(current_app.instance_path, 'word2vec')
        train_class=Train(word_model_dir)
        train_class.main(data['directory'],current_app.instance_path,word_model_dir,model_name)
        return jsonify(Response(msg='text cnn model train complete').__dict__)
    except Exception as e:
        current_app.logger.errot(e)
        return jsonify(Response(code=-1, msg='service call error').__dict__)


@bp.route('/predict/local/top',methods=['POST'])
def predict_cnn():
    """
    参数格式
    {
        {
            "name":"predict_local_top",
            "data":{
                "name":"textcnn",
                "top":3,
                "data":"D:/textcnn/dataset/test",
            }
        }
    }
    :return:
    """
    try:
        param = json.loads(request.get_data())
        if param['name'] != 'predict_local_top':
            return jsonify(Response(code=1, msg='service interface name error').__dict__)
        data = param['data']
        # categories_path=param['categories_path']
        model_name = data['name']
        checkpoint_dir=os.path.join(current_app.instance_path, 'runs',model_name)
        categories=list(open(os.path.join(checkpoint_dir, 'categories.txt'),"r",encoding='utf-8').readlines())
        categoriesArray=categories[0].split(',')
        if len(categoriesArray) < int(data['top']):
            msg = 'the number of model classifications is {}'.format(len(categoriesArray))
            return jsonify(Response(code=-1, data=msg, msg='error').__dict__)
        else:
            top_num=data['top']
            testdata_dir=data['data']
            predictlocalclass=Predict()
            ruseltdata=predictlocalclass.predict(checkpoint_dir,testdata_dir,top_num,categoriesArray)
            return jsonify(Response(data=ruseltdata, msg='service call 成功').__dict__)
        # predictlocalclass.predict(categories_path, checkpoint_filepath, testdata_dirpath)
        # predictlocalclass.predict(mpdel_name, top_num, testdata_dir)
        # return jsonify(Response(data=ruseltdata, msg='service call 成功').__dict__)
    except Exception as e:
        current_app.logger.errot(e)
        return jsonify(Response(code=-1, msg='service call error').__dict__)


@bp.route('/predict/real/top',methods=['POST'])
def predict_real_top():
    """
    参数格式
        {
            "name":"predict_real_top",
            "data":{
                "name":"textcnn",
                "top":3,
                "data":"nichola nicklebi celabr human spirit unrel dickensian decenc turn me horror scroog",
            }
        }
    :return:
    """
    try:
        param = json.loads(request.get_data())
        if param['name'] != 'predict_real_top':
            return jsonify(Response(code=1, msg='service interface name error').__dict__)
        data = param['data']
        model_name = data['name']
        checkpoint_dir=os.path.join(current_app.instance_path, 'runs',model_name)
        categories=list(open(os.path.join(checkpoint_dir, 'categories.txt'),"r",encoding='utf-8').readlines())
        categories_array=categories[0].split(',')
        if len(categories_array) < int(data['top']):
            msg = 'the number of model classifications is {}'.format(len(categories_array))
            return jsonify(Response(code=-1, data=msg, msg='error').__dict__)
        else:
            predict_content=data['data']
            top_num=data['top']
            predict_real_class=Predict_Real()
            ruselt_data=predict_real_class.predict(checkpoint_dir,predict_content,top_num,categories_array)
            return jsonify(Response(data=ruselt_data, msg='service call 成功').__dict__)
    except Exception as e:
        current_app.logger.errot(e)
        return jsonify(Response(code=-1, msg='service call error').__dict__)
