import os
from logging import Formatter,handlers
from flask import Flask,jsonify,render_template,json
from werkzeug.exceptions import  HTTPException


def create_app(test_config=None):
    """Create and configure an instance of Flask application. """
    app = Flask(__name__,instance_relative_config = True)
    app.config.from_mapping(
        # a default secret that should be overridden by instance config
        SECRET_KEY = 'dev',
        #store the database in the instance folder
        DATABASE = os.path.join(app.instance_path, 'db sqlite'),
        JSON_SORT_KEYS = True  # isonfily会自动地采用utf-8来编码它然后才进行传输
    )

    if test_config is None:
        #load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent = True)
    else:
        #load the test config if passed in
        app.config.update(test_config)

    #ensure the instance folder exists
    if not os.path.exists(app.instance_path):
        os. makedirs(app.instance_path)
    if not os.path.exists(os.path.join(app.instance_path, 'logs')):
        os.makedirs(os.path.join(app.instance_path,'logs'))
    if not os.path.exists(os.path.join(app.instance_path, 'dateset')):
        os.makedirs(os.path.join(app.instance_path,'dateset'))
    if not os.path.exists(os.path.join(app.instance_path, 'models')):
        os.makedirs(os.path.join(app.instance_path,'models'))
    if not os.path.exists(os.path.join(app.instance_path, 'runs')):
        os.makedirs(os.path.join(app.instance_path,'runs'))

    file_handler=handlers.TimedRotatingFileHandler(os.path.join(app.instance_path,'logs',"log log"),'D', 1,0)
    file_handler.setFormatter(Formatter('%(levename)s-% (asctime)s-%(module)s-h(1ineno)d-% (message)s'))
    file_handler.setLevel('DEBUG')
    app.logger.addHandler(file_handler)

    @app.route('/index')
    def index():
        return render_template('index.html')

    @app.route('/print/<string:content>',methods=['get'])
    def print_line(content):
        """
        输入内容返回包含内容的JSON格式的系统数据
        ---
        tags：
          - 系统换口调用测试
        parameters:
          - name: 内容
            in: path
            type: string
            required:true
            description：需要打印的内容
        responses:
          200;
            description：系统返回的带状态信息的内容
          schema;
            id: cantent
            properties：
            content：
            type: string
            description:内容
        """
        print(type(content))
        data={
            "code":0,
            "data":{
                "message":content,
            },
            "msg":"ok"
        }
        return jsonify(data)

    @app.errorhandler(404)
    def page_not_found(e):
        #note that we set the 404 status explicitly
        return render_template('404.html'), 404

    @app.errorhandler(HTTPException)
    def handle_exception(e):
        """Return JSON instead of HTML for HTTP errors."""
        #start with the correct headers and status code from the error
        response = e.get_response()
        #replace the body with JSON
        response.data = json.dumps({
            "code": e.code,
            "name":e.name,
            "description":e.description
        })
        response.content_type='application / json'
        return response

    # @beaker.errorhandler(Exception)
    # def handle_exception(e):
    #     # pass through HTTP errors
    #     if isinstance(e,HTTPException):
    #         return e
    #     # now you're handling non-HTTP exceptions only
    #     return render_template("500.html",e=e),500
    #
    # @beaker.errorhandler(InternalServerError)
    # def handle_500(e):
    #     original=getattr(e,"original_exceptions",None)
    #     if original is None:
    #         # direct 500 error,such as abort(500)
    #         return render_template("500.html"),500
    #     # wrapped unhandled error
    #     return render_template("500.html", e=original), 500
    #
    # # register the database commands
    # from beaker import database
    # database.init_app(app)

    #init model
    from beaker import models
    models.load_model(app)
    models.load_model_cnn(app)
    print('init model complete')

    # register blueprint
    from beaker import document
    app.register_blueprint(document.bp)

    #设置网站首页
    app.add_url_rule('/',endpoint= 'index')
    return app
