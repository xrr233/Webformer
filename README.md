# Webformer
Source code of SIGIR2022 Long Paper:

[Webformer: Pre-training with Web Pages for Information Retrieval](https://dl.acm.org/doi/abs/10.1145/3477495.3532086)

## Pipeline

### Preinstallation
First, prepare a **Python3** environment, and run the following commands:
```
  git clone https://github.com/xrr233/Webformer.git Webformer
  cd Webformer
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Prepare the Corpus Data
Every piece of corpus data is the raw HTML code of a web page.
Then run the following commands to clear irrelevant content and get the training corpus:
```
  python Preprocess/html2json.py 
  python 
```
