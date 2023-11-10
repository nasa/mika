from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="nasa-mika",
    version="1.0.2",
    author="Hannah Walsh and Sequoia Andrade",
    author_email="hannah.s.walsh@nasa.gov",
    maintainer_email= "sequoia.r.andrade@nasa.gov",
    maintainer= "sequoiarose",
    description="Manager for Intelligent Knowledge Access (MIKA)",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/nasa/mika.git",
    download_url="https://github.com/nasa/mika/archive/refs/tags/v1.0.2.tar.gz",
    project_urls = {"Documentation": "https://nasa.github.io/mika/"},
    keywords=["Natural Language Processing", "Knowledge Management", "Topic Modeling"],
    packages=find_packages(exclude=["test", "examples", "docs", "data"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved",
        'Development Status :: 3 - Alpha'
        ],
    python_requires='>=3.8',
    install_requires=[
        'BERTopic', 'datasets', 'gensim', 'matplotlib', 'nltk', 'numpy', 'octis', 'pandas',
        'pathlib', 'pingouin', 'pyLDAvis', 'regex', 'scikit-learn', 'scipy', 'seaborn',
        'sentence-transformers', 'spacy', 'symspellpy', 'tomotopy', 'transformers', 'wordcloud'
        ]
)