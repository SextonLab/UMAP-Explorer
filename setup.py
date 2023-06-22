from importlib.metadata import entry_points
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

import os

if os.name =='nt':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

setup(
    name='umapexplore',
    version='0.01',
    description='Set of tools and notebooks for UMAP exploration',
    long_description=readme(),
    classifiers=[
        'Development Status :: Number - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8'
    ],
    keywords='datascience cellprofiler',
    url='https://github.com/SextonLab/UMAP-Explorer',
    author='bhalliga',
    author_email='bhalliga@med.umich.edu',
    license='MIT',
    packages=['umapexplore'],
    entry_points = {
        # 'console_scripts' : []
    },
    install_requries=[
        'sqlite3', 'umap-learn'
    ],
    dependency_links=[],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
)