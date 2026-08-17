# writefile rfc_prototype/setup.py

from setuptools import setup, find_packages

setup(
    name='rfc_prototype',
    version='0.1.0',
    packages=find_packages(where='.', include=['rfc_prototype*']),
    package_dir={'': '.'}, # Ensure rfc_prototype is recognized at the top level
    include_package_data=True,
    install_requires=[
        'rebound',
        'scikit-learn==1.6.1', # Pin version for reproducibility
        'numpy>=1.19.5',
        'joblib>=1.2.0',
        'pandas'
    ],
    package_data={
        'rfc_prototype': ['models/*.joblib', 'data/*.csv']
    },
    description='Random Forest Classifier Prototype for Planet Stability',
    author='Jarrod Bieber',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/jarrodsb/ETAMU-binary-systems/rfc_prototype', # My GitHub URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
