# import io
#
# from setuptools import find_packages
# from setuptools import setup
#
# with io.open("README.md","rt",encoding="utf-8") as f:
#     readme=f.read()
#
# setup(
#     name="beaker",
#     version="0.1.0",
#     #url="http://127.0.0.1:5000"
#     license="BSD",
#     maintainer="brouka",
#     maintainer_email="brouka@163.com",
#     description="this is a model service project",
#     long_description=readme,
#     packages=find_packages(),
#     include_package_data=True,
#     zip_safe=False,
#     install_requires=[
#         "scikit-Learn>=0.19.1",
#         "scipy>=1.1.0",
#         "numpy>=1.18.1",
#         "Flask>=1.1.1",
#         "gensim>=3.7.2"
#     ],
#     extras_require={"test":["pytest","coverage"]},
# )
