# Big Data Assignment 2
Alla Chepurova, DS-01
Amina Miftahova, DS-01
Lev Svalov, DS-02
Maxim Evgrafov, DS-01

## Introduction
Sentiment analysis is a data mining task that aimed to solve the problem of determining emotional/subjective/opinion coloring of the data such one from social medias. It uses natural language processing and machine learning techniques to interpret and classify emotions in subjective information.

The aim of this assignment was to perform sentiment analisys in a stream-fashion way using Apache Spark, Scala, NLP and machine learning techniques.


## Overview
The pipeline in our implementation is exactly the same as in the assignment description.

![](https://i.imgur.com/dYRljAb.png)
([image source](https://hackmd.io/@BigDataInnopolis/ryWL1HiKS))

### Algorithm
First of all, we train the models used for preprocesing and prediction and save them for the further usage. The model needed are: 
* **Word2Vec** for converting tweets into the array of numerical features
* **Logistic Regression** as the simplest and the most well-known parametrical model for binary classification
* **Random Forest** as a relatively simple, but a promising model.
* **K-Nearest Neighbours** as a model for our own implementation (surprisingly not presented in `spark.ml`)

The second step is evaluation of the models on the test set.

And, finally, we accepted the tweets from the stream and run them through the prediction pipeline, where they are processed and as a result written into the file for the further model evaluation. This final evaluation was done manually using the most modern and powerful tool for analysis (*Excel*).
### Dataset
We have decided to use the [Twitter Sentiment Dataset](https://www.kaggle.com/c/twitter-sentiment-analysis2/data), as it seems to be most suitable for the sentiment analysis on data coming from Twitter.

## Preprocessing
The preprocessing consists of two main steps: filtering and vectorization.
### Filtering
The data in the `train.csv` file contains tweet text and their sentiment. Example of the row in the file:
```
4,0,          .. Omgaga. Im sooo  im gunna CRy. I've been at this dentist since 11.. I was suposed 2 just get a crown put on (30mins)...
```
The columns are:
```
ItemID, Sentiment, Text
```
After we read the `.csv` dataset, we perform the text clearing. It consists of:
1. Stop words removal
2. HTML-tags removal
3. Links removal
4. Aliases removal and adding a column-indicator for the alias found
5. Character repetition removal and adding a column-indicator for the repetition found
6. Tokenization and convertation to lowercase
7. Adding a column for an exclamation mark found

Steps 3-7 were done using regex patterns, for the step 1 we have used **RegexTokenizer** and for the step 2 **StopWordsRemover**, both from `spark.ml` library.
### Vectorization
The vectorization method chosen was Scala **Word2Vec** Model. The model is trained on the train dataframe. The parameters we have changes are:

```
vecSize = 30
minCount = 10
```
Thus, each tweet is represented by 30-dimensional vector, and for the word to make a contribution into that feature vector it had to occure at least 10 times in the train dataset.

The dataframe returned after the preprocessing step is of the following form:

```
+------+------------------+--------+------------+-------------+
|  id  |     filtered     | alias  | repetition | exclamation |
+------+------------------+--------+------------+-------------+
|      | array of strings | 0 or 1 |   0 or 1   |   0 or 1    |
+------+------------------+--------+------------+-------------+
```


## Logistic Regression
As the first model we have decided to use **Logistic Regression** as the most well-known and easy to use parametric model for binary classification.

The logistic regression model used is presented below.
```
org.apache.spark.ml.classification.LogisticRegression
```
We have performed the hyperparameters tuning using GridSearch method. The parameters the search was performed on are: `regParam`, `threshold` and`fitIntercept`. The best parameters obtained were the following:
```
regParam = 0.0
threshold = 0.5
fitIntercept = true
```

**The best f1 score:** 0.73
**The threshold for the best perfomance:** 0.5

## Random Forest
**Random forest** is an ensemble learning method for classification and regression. Random decision forests are aimed to correct the decision trees' habit of overfitting to their training set, so that is why we preferred this model to simple decision tree one. 

To build a model we used  functions from 
```
org.apache.spark.ml.regression.RandomForestClassificationModel
``` 
package. 
Also, we have performed the hyperparameters tuning using GridSearch method. The parameters the search was performed on are: `maxDepth`, `threshold` and`numTrees`. The best parameters obtained were the following:
```
maxDepth = 10
threshold = 0.3
numTrees = 30
```

**The best f1 score:** 0.72
**The threshold for the best perfomance:** 0.3

## KNN
The algorithm we decided to implement by our own way is a variation of KNN (K nearest neigbours) model. In KNN algorithm there there is a need to calculate a distance between the point we classifying and the whole dataset, then we should take first k (which is a hyper parameter) "nearest" points and assign the most frequent label among them to the point we are classifying.
As it was said in the preprocessing section, we used word2vec model to get vector representation of every tweet in the corpus to use those as features in our models further. This representation was very handful as we were able to calculate the distance between tweets as a distance between corresponding vectors using any metrics. The most appropriate one to implement on scala vectors was *cosine similarity*. 
![](https://i.imgur.com/sfoX6cb.png)
After calculating the distances we sorted them in ascending order and took first k neigbours. So that, there remains the most similar tweets to the target one. And the label we assign to new tweet is the most frequent label among those k points.

## Stream handler

**Data Stream** is a continuous flow of the information. As was stated in the assigment description, stream could be accessible by ("10.90.138.32", 8989). 

So, what is the main parts of the stream processing? First of all, we should initialize stream reading process. Then we assign computations that we apply to stream of data and show the output of the resulting computation. At very end we start reading a stream and denote termination conditions.

Let's take a look on the code:



---

**Initializing Stream**

```scala
val tweets = spark.readStream
                  .format("socket")
                  .option("host", "10.90.138.32")
                  .option("port", "8989")
                  .load()
```
Here we denote that the source of the stream is a *socket* with such *host* and *port*. 
*readStream* returns *DataFrame* object that represents data like table. Raw data from the socket (variable *tweets*) will be in this format:
```
+----------+
|   value  |
+----------+
|tweet_text|
+----------+
```
---
**Computations stage**
```scala
val notNull = tweets.filter("value != ''")

val Models_predict = notNull.select(col("value"),
      lit(predict(col("value").toString(), preprocessor, models, w2vModel, lrModel, sc))
              .as("1st model"),
      lit(predict(col("value").toString(), preprocessor, models, w2vModel, rfModel, sc))
              .as("2nd model"))
```
With the first command we *filter* all values, because we should use tweets with *any text inside*. Shortly, we use value that are not empty.
Secondly, we *predict the label* of the tweet for *each model* and add this predictions to DataFrame.

---
**Output Stage**
```scala
val query = Models_predict.writeStream
                        .format("csv")
                        .option("format", "append")
                        .option("path", "hdfs://namenode:9000/user/sharpei/output/")
                        .option("checkpointLocation", "hdfs://namenode:9000/user/sharpei/checkpoint_dir/")
                        .outputMode("append")
                        .start()
                        .awaitTermination(60000 * 5)
```

Here we describe that format of the output will be *.csv* and script should put it inside *"path"* directory. We start stream processing and denote termination conditions as termination *from user* or *timeout* which is 60000 * 5 ms (5 minutes).

Because of the fact that there will be only *one tweet per a message* we will see a lot of small files inside "path" dir and every of them will contains information *only about one tweet*. That is why we added additional command that performs *merge* of this files and put final result inside *"finaloutput"* directory.

```scala
spark.read
    .csv("hdfs://namenode:9000/user/sharpei/output/")
    .coalesce(1).write
    .csv("hdfs://namenode:9000/user/sharpei/final_output/")
```
---

## Evaluation and comparison
### Evaluation
To find **the best hyperparameters** we used methods from packages ```rg.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}```. For each model we specifyed corresponding set of parameters we tune using grid search object and estimate the best values of them  through cross validation.

The **performance of models** was evaluated using [F1-score](https://en.wikipedia.org/wiki/F1_score). 
Firstly, we have used the f1-score evaluator from
```
BinaryClassificationMetrics
```
which calculate f1 score for each threshold and returns the **maximum f1 score and corresponding threshold**.  
However, we were not sure on the correctness of its work and in order to double-check it decided to implement our custom function, which calculates precision and recall and that based on them, f1-score. This function provided us with another advantage for the models comparison and analysis of the performance.
### Comparison

#### Test set

Unfortunately, we did not succeed in making the KNN work on the test set due to that weird error: `org.apache.spark.SparkException: Task not serializable`, thus, we are not able to compare it to the other two algorithms.
```
+----------+---------------------+---------------+
|          | Logistic Regression | Random Forest |
+----------+---------------------+---------------+
|precision |        0.58         |      0.56     |
+----------+---------------------+---------------+
|recall    |        0.99         |      0.99     |
+----------+---------------------+---------------+
|f1 score  |        0.73         |      0.72     |
+----------+---------------------+---------------+

```
As we can see, both model do not actually perform as well as we wished to. The recall is good, but the precision is a weak point for both of them.
The solution is to adjust the threshold, however, that will affect the recall badly.

The other, more robust and elegant solution is to change the train-test dataset. The dataset is not balanced, as well as the function we use for train-test split does not allow to save the distribution of the classes.

#### Stream data

The stream tweets and predictions can be found in `tweets_classified.csv` file. As you could observe, almost no of them have a strong negative sentiment. And due to our models having relatively low threshold almost all of the tweets were classified as positive.
```
+----------+---------------------+---------------+------+
|          | Logistic Regression | Random Forest |  KNN |
+----------+---------------------+---------------+------+
|precision |        0.93         |      0.94     | 0.92 |
+----------+---------------------+---------------+------+
|recall    |          1          |       1       | 0.69 |
+----------+---------------------+---------------+------+
|f1 score  |        0.96         |      0.97     | 0.79 |
+----------+---------------------+---------------+------+
```

NOTE: the `tweets_classified.csv` has no timestamps because of our inattention. Later, we have fixed that and the final version of how the output file looks like may be observed in `tweets_final.csv`.

## How to execute

```
spark-submit --master yarn <path to jar> <path to train.csv>
```

In our github, the `jar` executable is under the `scala_app/target/scala-2.12/assignment2_2.12-1.0.jar` path.

**VERY IMPORTANT**
For the application to use pretrained models, put all directories from the `models` directory in our github to the home folder of HDFS (`hdfs://namenode:9000/user/sharpei/`).

The output of the stream data predictions is in the `final_output` folder in the home directory of HDFS.


## Conclusion
The team has achieved the main project goal, and performed sentiment analisys of tweets  in a stream-fashion way.
Moreover, as a learning result of accomplishing of the project, team members:
* Practiced competencies of:
    * Scala programming
    * Text processing
    * Working with stream data
* Acquired knowledge of the following areas:
    * NLP
    * Machine learning
    * Functional programming

### Possible improvements
Our implementation is 🙂
But it could be 😎

As in every project that deals with the Machine Learning algorithms, we have an endless amount of improvements that might be introduced to our implementation, and here are only a small part of them:

1. **Normalize the train dataset**. For now, the train data is not balanced: the number of the tweets of one or the other class is not equal. What is more, it is hard to implement in Scala a train-test split that will preserve the initial distribution of class samples.
2. **Use stemming, lemmatization and dictionary lookups**. We were not able to find a working implementation of any of them on Scala, but it might improve the accuracy, especially, when the data is coming from twitter: it contains lots of spelling mistakes and acronyms.
3. **Use non-word features in the vectors**. For now we can use information about exclamation marks, reprtitive characters and aliases. Furthermore, we could parse smiles and somehow include these features into the tweet vector representation.
4. **Try out different vectorization techinques**. Now we use Word2Vec algorithm and did not tune the hyperparameters for it. One option is to tune them, and the other option is to experiment with the other methods, for example, TF/IDF or even something more advanced like BERT.
5. **Tune the algorithms**. As we can observe, the algorithms show a good recall and low precision. That might be fixed by tuning the threshold and adjusting the recall-precision balance using not f1-score, but f-score with the *beta* equal to something other that 1. Also, we did not perform the KNN tuning: changes in K and the formula used for distance might change the performance.

## Team members contribution
**Amina**:
* Preprocessing
* Logistic Regression Model
* Combining the parts into one pipeline

**Alla**:
* Preprocessing
* Random Forest Classifier
* KNN Classifier

**Maxim**:
* Stream data handling
* Stream data processing pipeline

**Lev**:
* Models evaluation and comparison

Note: not all members did contribute to the `master` branch, some of the work is only on the branch for the particular person.

## Link to Git repo
https://github.com/screemix/tweet_analyzer

## References
* https://spark.apache.org/docs/latest/streaming-programming-guide.html
* https://spark.apache.org/docs/latest/ml-features.html#tf-idf
* https://spark.apache.org/docs/latest/ml-classification-regression.html

