from setuptools import setup, find_packages

setup(
    name='qsoabsfind',
    use_scm_version=True,
    author='Abhijeet Anand',
    author_email='abhijeetanand2011@gmail.com',
    description='A Convolution-Based, Adaptive S/N Framework for Detecting Metal Doublet Absorption in Low-Resolution Quasar Spectra',

    packages=find_packages(),
    install_requires=[
        'numpy',
        'astropy',
        'scipy',
        'numba',
        'matplotlib',
        'tqdm',
        'pyyaml',
    ],
    extras_require={
        'dev': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'qsoabsfind=qsoabsfind.parallel_convolution:main',
        ],
    },
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/abhi0395/qsoabsfind',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)