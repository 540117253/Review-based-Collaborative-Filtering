# Review-based-Collaborative-Filtering
A model zoo that contains different review-based collaborative filtering models of rating prediction task for recommendation

## The Definition of Review-based rating prediction 
![image](https://github.com/540117253/Review-based-Collaborative-Filtering/blob/master/illustration/Task%20Definition.jpg)


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
According to the mapping method of review text, there are two strategies to preprocess the dataSet. The first one called wordVectorBased, which is to represent the review by word vectors: `wordVectorBased_data_preprocess.py`(/Model_Zoo/wordVectorBased_data_preprocess.py). The second one is bertBased that is to represent the review by BERT's embedding: `bertBased_data_preprocess.py`(/model_Zoo/bertBased_data_preprocess.py)
#### 2.2 Models Definition
##### 2.2.1 WordVectorBased Models
&ensp; | Model Name | Directory
---|---|---
1 | DeepCoNN | Model_Zoo/models/wordVectorBased_Models
2 | DARR | Model_Zoo/models/wordVectorBased_Models
3 | MACF | Model_Zoo/models/wordVectorBased_Models
##### 2.2.2 BertBased Models
&ensp; | Model Name | Directory
---|---|---
1 | DeepCLFM | Model_Zoo/models/bertBased_Models
2 | NCEM | Model_Zoo/models/bertBased_Models
#### 2.3 Runner
Since there are

基于词向量和基于bert的模型共分为两类，
它们的代码区别在于预处理阶段：词向量的有vocabulary，而基于bert的没有
运行的run.py中：词向量的有预训练词向量选项，而基于bert的没有

另外，基于bert的模型的预处理需要额外运行bert_service的服务。

