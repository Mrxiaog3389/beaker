class Model:
    def __init__(self):
        self.id=None
        self.name=None

class Request:
    def __init__(self,name,data=None):
        """
        :param name: 服务接口名称，用于接口验证
        :param data: 接口调用参数s
        """
        self.name=name
        self.data=data

class Response:
    def __init__(self,code=1,data=None,msg="ok"):
        """
        :param code: 0,正常，-1，错误
        :param data: 响应数据
        :param msg: "ok",正常时提示信息,"error":错误时提示信息
        """
        self.code=code
        self.data=data
        self.msg=msg