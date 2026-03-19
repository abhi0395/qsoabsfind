from setuptools import setup, find_packages

setup(
    name='qsoabsfind',
    use_scm_version=True,
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
        'pytest'
    ],
    entry_points={
        'console_scripts': [
            'qsoabsfind=qsoabsfind.parallel_convolution:main',
        ],
    },
    author='Abhijeet Anand',
    author_email='abhijeetanand2011@gmail.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/abhi0395/qsoabsfind',
)