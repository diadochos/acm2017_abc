from setuptools import setup

setup(
    name='pyabc',
    version='0.1',
    description='Implementation of different samplers for Applied Bayesian Computation (ABC)',
    url='https://github.com/compercept/acm2017_abc/tree/master/pyabc',
    author='Michael Burkhardt, Fabian Kessler, Dominik Straub',
    author_email='mbphoenix@t-online.de',
    license='MIT',
    packages=['pyabc'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib', 'emcee', 'GPyOpt'
    ],
    zip_safe=False
)
