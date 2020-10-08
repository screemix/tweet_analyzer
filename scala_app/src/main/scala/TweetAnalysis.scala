import org.apache.spark
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.sql.catalyst.dsl.expressions.StringToAttributeConversionHelper
import org.apache.spark.sql.functions._
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.Word2VecModel
import java.io.File
import java.time.format.DateTimeFormatter
import java.time.LocalDateTime

import preprocessor.Preprocessor
import models.Models

object TweetAnalysis {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val inputPath = args(0)

    val spark = SparkSession.builder().appName("very streaming very scala \uD80C\uDD8F").getOrCreate()
    val sc = spark.sparkContext
    val preprocessor = new Preprocessor()
    val models = new Models()

    // evaluate on train data
    val trainedModels = evalOnTrainTest(inputPath, preprocessor, models, sc)
    val w2vModel = trainedModels._1
    val lrModel = trainedModels._2
    val rfModel = trainedModels._3

    // stream analysis
    println("START STREAM PROCESSING")
    val tweets = spark.readStream.format("socket").option("host", "10.90.138.32").option("port", "8989").load()

    val notNull = tweets.filter("value != ''")

    // val Models_predict = notNull.withColumn("1st model", AllaModel.predict(col("value").toString()))
    // 							.withColumn("2nd model", AminaModel.predict(col("value").toString()))

    val formatter = DateTimeFormatter.ofPattern("MM-dd--HH_mm_ss")
    
    val Models_predict = notNull.select(lit(formatter.format(LocalDateTime.now())), col("value"),
      lit(predict(col("value").toString(), preprocessor, models, w2vModel, lrModel, sc)).as("1st model"),
      lit(predict(col("value").toString(), preprocessor, models, w2vModel, rfModel, sc)).as("2nd model"))


    val query = Models_predict.writeStream
      .format("csv")
      .option("format", "append")
      .option("path", "hdfs://namenode:9000/user/sharpei/output/")
      .option("checkpointLocation", "hdfs://namenode:9000/user/sharpei/checkpoint_dir/")
      .outputMode("append")
      .start()
      .awaitTermination(60000 * 30)


    spark.read.csv("hdfs://namenode:9000/user/sharpei/output/").coalesce(1).write.csv("hdfs://namenode:9000/user/sharpei/final_output/")

  }


  def predict(tweet: String, preprocessor: Preprocessor, models: Models, w2vModel: Word2VecModel, model: CrossValidatorModel, sc: SparkContext): Int = {
    val tweetCleared = preprocessor.clear_input(tweet, sc)
    val prediction = models.predict(tweetCleared, w2vModel, model, sc)
    prediction
  }


  def evalOnTrainTest(inputPath: String, preprocessor: Preprocessor, models: Models, sc: SparkContext): (Word2VecModel, CrossValidatorModel, CrossValidatorModel) = {

    // creates or loads models
    //

    val conf = sc.hadoopConfiguration
    val fs = org.apache.hadoop.fs.FileSystem.get(conf)

    // clear train file
    println("start preprocessing")
    val train_cleared = preprocessor.clear_train(inputPath, sc)

    // train-test split
    val Array(train, test) = train_cleared.randomSplit(Array[Double](0.7, 0.3))

    // ----------------W2V MODEL----------------
    var trainW2V = true
    if (fs.exists(new org.apache.hadoop.fs.Path("w2vModel"))) {
      println("loading existing word 2 vec model")
      trainW2V = false
    }

    if (trainW2V) {
      // train w2v model
      println("start training word2vec")
      val w2v_t = preprocessor.word2vec_train(train, 30, 10, "filtered", "features")
      w2v_t.save("w2vModel")
    }

    // load model from file
    val w2vModel = Word2VecModel.load("w2vModel")
    // ----------------W2V MODEL----------------



    // ----------------LOGREG MODEL----------------
    println("---------------LOGISTIC REGRESSION---------------")

    var trainLogreg = true
    if (fs.exists(new org.apache.hadoop.fs.Path("logregModel"))) {
      println("loading existing logistic regression model")
      trainLogreg = false
    }

    if (trainLogreg) {
      // train the model
      println("start training logreg")
      models.logregTrain(train, w2vModel, sc)
    }

    // load model from file
    val lrModel = CrossValidatorModel.load("logregModel")

    // evaluate on test data
    val scores_lr = models.testEval(test, w2vModel, lrModel, sc)

    println("SCORES ON TEST DATA")
    print("precision: ")
    println(scores_lr._1)
    print("recall: ")
    println(scores_lr._2)
    print("f1 score: ")
    println(scores_lr._3)
    // ----------------LOGREG MODEL----------------



    // ----------------RAND FOREST MODEL----------------
    println("---------------RANDOM FOREST---------------")

    var trainRF = true
    if (fs.exists(new org.apache.hadoop.fs.Path("rfModel"))) {
      println("loading existing random forest model")
      trainRF = false
    }

    if (trainRF) {
      // train the model
      println("start training random forest")
      models.rfTrain(train, w2vModel, sc)
    }
    // load model from file
    val rfModel = CrossValidatorModel.load("rfModel")

    // evaluate on test data
    val scores_rf = models.testEval(test, w2vModel, rfModel, sc)

    println("SCORES ON TEST DATA")
    print("precision: ")
    println(scores_rf._1)
    print("recall: ")
    println(scores_rf._2)
    print("f1 score: ")
    println(scores_rf._3)
    // ----------------RAND FOREST MODEL----------------

    (w2vModel, lrModel, rfModel)
  }
}
