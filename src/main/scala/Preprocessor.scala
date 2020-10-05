import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

class Preprocessor {

  def clear(file_path: String, sc: SparkContext) {
    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    val train_data_file = sc.textFile(file_path)

    val train_data_unsliced = train_data_file.map(line => (line.split(",")(0), line.split(",")(1), line.split(",")(2))).collect()

    val train_data = sc.parallelize(train_data_unsliced.slice(1, train_data_unsliced.length))
      .map(x => (x._1.toInt, x._2.toInt, x._3)).toDF("id", "sentiment", "sentence")

    val regexTokenizer = new RegexTokenizer().
      setInputCol("sentence").
      setOutputCol("words").
      setPattern("\\w+").setGaps(false)

    val train_tokenized = regexTokenizer.transform(train_data)

    val remover = new StopWordsRemover().
      setInputCol("words").
      setOutputCol("filtered")

    val train_cleared = remover.transform(train_tokenized).select("id", "sentiment", "filtered")
    train_cleared
  }

  def tfIdf(data: sql.DataFrame): Unit ={
    val zero_sent = data.filter("sentiment == 0")
    val  one_sent = data.filter("sentiment == 1")

    val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("TF").setNumFeatures(20)
    val tfDataZeroSent = hashingTF.transform(zero_sent)
    val tfDataOneSent = hashingTF.transform(one_sent)

    val idf = new IDF().setInputCol("TF").setOutputCol("TFIDF")
    val idfModelZero = idf.fit(tfDataZeroSent)
    val idfModelOne = idf.fit(tfDataOneSent)
    val rescaledZero = idfModelZero.transform(tfDataZeroSent).select("id", "sentiment", "filtered", "TFIDF")
    val rescaledOne = idfModelOne.transform(tfDataOneSent).select("id", "sentiment", "filtered", "TFIDF")
    val rescaled = rescaledOne.union(rescaledZero)
    rescaled
  }

  def word2Vec(train_cleared: sql.DataFrame): Unit = {
    val word2Vec = new Word2Vec().setInputCol("filtered").setOutputCol("vec").setVectorSize(3).setMinCount(0)
    val model = word2Vec.fit(train_cleared)
    val result = model.transform(train_cleared)
    result
  }
}

  def randForest(result: sql.DataFrame): Unit = {
    val rf = new RandomForestRegressor().setLabelCol("sentiment").setFeaturesCol("vec")
    val model = rf.fit(result)
    val predictions = model.transform(result)
    val preds = predictions.withColumn("rounded_prediction", round(col("prediction")))
    val evaluator = new RegressionEvaluator().setLabelCol("sentiment").setPredictionCol("rounded_prediction").setMetricName("rmse")
    val rmse = evaluator.evaluate(preds)
    rmse
  }

class clearInput {
  def classify(twit: String): Unit ={
    val processed_tweet = tweet.replaceAll("[''()_,!?;;]", "").trim.toLowerCase
    val tweet_df = Seq(processed_tweet).toDF("sentence")
    val regexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("words").setPattern("\\w+").setGaps(false)
    val tweet_tokenized = regexTokenizer.transform(tweet_df)
val result = model.transform(train_cleared)
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val tweet_cleared = remover.transform(tweet_tokenized).select("sentence", "filtered")

    val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("TF").setNumFeatures(5)
    val tweetTF = hashingTF.transform(tweet_cleared)
    tweetTF

  }
}