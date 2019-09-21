# Review-based-Collaborative-Filtering
A model zoo that contains different review-based collaborative filtering models of rating prediction task for recommendation

# The Definition of Review-based rating prediction 


# 1. Model Type
### 1.1 WordVectorBased Models
&ensp; | Model Name
---|---
1 | DeepCoNN
2 | DARR
3 | MACF

### 1.2 BertBased Models
&ensp; | Model Name
---|---
1 | DeepCLFM
2 | NCEM

# 2. Code Description
### 2.1 DataSet and Preprocess



基于词向量和基于bert的模型共分为两类，
它们的代码区别在于预处理阶段：词向量的有vocabulary，而基于bert的没有
运行的run.py中：词向量的有预训练词向量选项，而基于bert的没有

另外，基于bert的模型的预处理需要额外运行bert_service的服务。

