package models

import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD.fromRDD
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD

class CustomKnn{
  def fit(data: DataFrame): Unit ={
    val data_vec = w2vModel.transform(data)
    data_vec
  }
  def predict(data: DataFrame, tweet: DataFrame, w2vModel: Word2VecModel, sc: SparkContext, k: Int): Int ={
    val tweet_vec = tweet.first().getAs[Vector]("vec")
    val dist = data
      .map(x => (math abs(x.getAs[Vector]("vec").dot(tweet_vec)) / (math sqrt(x.getAs[Vector]("vec").dot(x.getAs[Vector]("vec")) * tweet_vec.dot(tweet_vec)))))
    val dist_temp = dist.withColumn("rowId1", monotonically_increasing_id())
    val data_temp = data.withColumn("rowId2", monotonically_increasing_id())
    val distances = dist_temp.as("df1").join(data_temp.as("df2"), res3("rowId1") === res("rowId2"), "inner").select("df1.value", "df2.id", "df2.sentiment")

    val num =  distances.sort(col("value")).select(col("sentiment")).head(k).map(x => x(0).asInstanceOf[Int]).reduce(_+_)
    if (num >= k /2){
      1
    }
    else{
      0
    }

  }
}