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
Run the following commands to clear irrelevant content and get the training corpus:
```
  python Preprocess/html2json.py 
```
Remember to set your data path in the code.

### Prepare the Training Data
Use the json file output in the previous step to generate training data.
```
  bash construct_data.sh
```

### Running Pre-training

```
 bash train.sh
```

## Citations
If you use the code, please cite the following paper:  

```
@inproceedings{DBLP:conf/sigir/GuoMMQZJCD22,
  author    = {Yu Guo and
               Zhengyi Ma and
               Jiaxin Mao and
               Hongjin Qian and
               Xinyu Zhang and
               Hao Jiang and
               Zhao Cao and
               Zhicheng Dou},
  editor    = {Enrique Amig{\'{o}} and
               Pablo Castells and
               Julio Gonzalo and
               Ben Carterette and
               J. Shane Culpepper and
               Gabriella Kazai},
  title     = {Webformer: Pre-training with Web Pages for Information Retrieval},
  booktitle = {{SIGIR} '22: The 45th International {ACM} {SIGIR} Conference on Research
               and Development in Information Retrieval, Madrid, Spain, July 11 -
               15, 2022},
  pages     = {1502--1512},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3477495.3532086},
  doi       = {10.1145/3477495.3532086},
  timestamp = {Sat, 09 Jul 2022 09:25:34 +0200},
  biburl    = {https://dblp.org/rec/conf/sigir/GuoMMQZJCD22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


```
