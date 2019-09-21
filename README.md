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
If you run the `WordVectorBased Models`, first to run the `wordVectorBased_data_preprocess.py`, and then run the `run_wordVectorBased_model.py`.

If you run the `BertBased Models`, first to make sure the `bert_service` is started. Run the `bertBased_data_preprocess.py`, and then run the `run_bertBased_model.py`. 


