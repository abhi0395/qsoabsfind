from setuptools import setup, find_packages
import setuptools_scm

setup(
    name='qsoabsfind',
    use_scm_version=True, # Automatically detects version from Git tag
    setup_requires=["setuptools_scm"]
    description='A module to search for MgII/CIV absorption features in QSO spectra using convolution based adaptive S/N approach',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'astropy',
        'scipy',
        'numba',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'qsoabsfind=qsoabsfind.parallel_convolution:main',
        ],
    },
    package_data={
        'qsoabsfind': ['*.py'],
    },
    author='Abhijeet Anand',
    author_email='abhijeetanand2011@gmail.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/abhi0395/qsoabsfind',
)
