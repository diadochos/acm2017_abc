from setuptools import setup

setup(
    name='abc',
    version='0.1',
    description='Implementation of different samplers for Applied Bayesian Computation (ABC)',
    url='https://github.com/compercept/acm2017_abc/tree/master/abc',
    author='Michael Burkhardt, Fabian Kessler, Dominik Straub',
    author_email='mbphoenix@t-online.de',
    license='MIT',
    packages=['abc'],
    install_requires=[
        'numpy',
    ],
    zip_safe=False
)