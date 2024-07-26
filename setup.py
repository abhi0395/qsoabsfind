from setuptools import setup, find_packages

setup(
    name='qsoabsfind',
    version='0.1.0',
    description='A package to find QSO absorbers using convolution methods',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'astropy',
        'scipy',
        'fitsio',
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
    author_email='AbhijeetAnand@lbl.gov',
    description='A module to search for MgII/CIV absorption features in QSO spectra.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/abhi0395/qsoabsfind',
)
