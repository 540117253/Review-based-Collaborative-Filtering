# Review-based-Collaborative-Filtering
A model zoo that contains different review-based collaborative filtering models of rating prediction task for recommendation

### Definition of Review-based  Rating Prediction 
![image](https://github.com/540117253/Review-based-Collaborative-Filtering/blob/master/illustration/Task%20Definition.jpg)

### Requirements
* Python 3.6
* Tensorflow 1.13.1

## 1. Models
#### 1.1 WordVectorBased Models
&ensp; | Model Name
---|---
1 | DeepCoNN
2 | DARR
3 | MACF
#### 1.2 BertBased Models
&ensp; | Model Name
---|---
1 | DeepCLFM
2 | NCEM


## 2. Code Description
#### 2.1 DataSet and DataPreprocess
The experimental dataset is Amazon Review DataSet, which can be downloaded [here](http://jmcauley.ucsd.edu/data/amazon/links.html).

The pre-trained word vector is glove.6B, which can be downloaded [here](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.6B.zip).

According to the mapping method of review text, there are two strategies to preprocess the dataSet. 

&ensp; | DataPreprocessor | Description | Directory
---|---|---|---
1 | `wordVectorBased_data_preprocess.py` | split train, valid and test dataSet. Represent the review by word vectors. You may choose different dataSet and modify the preprocess params in the first serveral lines of this code | /Model_Zoo
2 | `bertBased_data_preprocess.py` | split train, valid and test dataSet. Represent the review by BERT's embedding. You may choose different dataSet and modify the preprocess params in the first serveral lines of this code | /Model_Zoo

#### 2.2 Models Definition
##### 2.2.1 WordVectorBased Models
&ensp; | Model Name | Directory
---|---|---
1 | `DeepCoNN.py` | Model_Zoo/models/wordVectorBased_Models
2 | `DARR.py` | Model_Zoo/models/wordVectorBased_Models
3 | `MACF.py` | Model_Zoo/models/wordVectorBased_Models
##### 2.2.2 BertBased Models
&ensp; | Model Name | Directory
---|---|---
1 | `DeepCLFM.py` | Model_Zoo/models/bertBased_Models
2 | `NCEM.py` | Model_Zoo/models/bertBased_Models
#### 2.3 Runner
&ensp; | Runner | Description | Directory
---|---|---|---
1 | `run_wordVectorBased_model.py` | rnn the wordVectorBased models, you may choose the specific model by commenting out codes nearly line 478 | /Model_Zoo
2 | `run_bertBased_model.py` | rnn the wordVectorBased models, you may choose the specific model by commenting out codes nearly line 425 | /Model_Zoo

## 3. How to Use
1. Make sure the glove pre-trained word vector ([download](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.6B.zip)).
and Amazon Review DataSet ([download](http://jmcauley.ucsd.edu/data/amazon/links.html)) in the following directory:
- `/data/`
  - `Automotive_5`
    - `Automotive_5.json`
  - `glove.6B`
    - `glove.6B.50d.txt`
    - `glove.6B.100d.txt`
    - `glove.6B.200d.txt`
    - `glove.6B.300d.txt`
    
2. If you run the `WordVectorBased Models`, first to run the `wordVectorBased_data_preprocess.py`, and then run the `run_wordVectorBased_model.py`.

3. If you run the `BertBased Models`, first to make sure the `bert_service` is started (How to start `bert_service` can read [here](https://github.com/hanxiao/bert-as-service)). Run the `bertBased_data_preprocess.py` (My bert_service machine's IP is 10.1.63.214, so you can change the IP of your bert_service machine in the first serveral lines of this code), and then run the `run_bertBased_model.py`. 


