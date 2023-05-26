from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="mika",
    version="1.0.0",
    author="Hannah Walsh and Sequoia Andrade",
    author_email="hannah.s.walsh@nasa.gov; sequoia.r.andrade@nasa.gov",
    description="Manager for Intelligent Knowledge Access (MIKA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nasa/mika.git",
    packages=find_packages(exclude=["test"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: ",
        "Operating System :: Linux and Windows",
        'Development Status :: 3 - Alpha'
        ],
    python_requires='>=3',
    install_requires=[
        'BERTopic', 'datasets', 'gensim', 'matplotlib', 'nltk', 'numpy', 'octis', 'pandas',
        'pathlib', 'pingouin', 'pkg_resources', 'pyLDAvis', 'regex', 'scikit-learn', 'scipy', 'seaborn',
        'sentence-transformers', 'spacy', 'symspellpy', 'tomotopy', 'torch', 'transformers', 'wordcloud'
        ]
)