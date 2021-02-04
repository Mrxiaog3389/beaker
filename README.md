## 服务通用数据说明
### 服务请求数据格式

{
    "name":0,  # 服务接口名称，用于接口验证
    "data":{...} # 接口调用参数
}
'''
### 服务响应数据格式
{
    "code":0,  # 0:正常，-1：错误
    "data":{...} # 响应数据
    "msg":"ok"   # "ok":正常时提示信息，"error":错误时提示信息
}
'''


## 项目相关命令
### 指定IP和Port
'''
python -m flask run -h 0.0.0.0 -p 8081
'''
### 生成依赖文件requirements.txt
'''
#安装
pip install pipreqs
#执行
pipreqs . --encoding=utf8  --force
'''
### 打包
'''
>python setup.py bdist_wheel
'''
### 安装并完成项目初始化
'''

>call venv\Scripts\activate
>pip install beaker-0.1.0-py3-none-any.whl
>set FLASK_APP=beaker
>flask init-db
'''
## 项目启动
'''


## 项目启动
'''
>venv\Scripts\activate
>set FLASK_APP=beaker
>flask run
'''