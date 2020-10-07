import org.apache.spark
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.sql.catalyst.dsl.expressions.StringToAttributeConversionHelper
import org.apache.spark.sql.functions._
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.Word2VecModel
import java.io.File

import preprocessor.Preprocessor
import models.Models

object TweetAnalysis {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("appName")
    val sc = new SparkContext(conf)

    evalOnTrainTest(sc)

  }

  def evalOnTrainTest(sc: SparkContext): Unit = {
    val inputPath = "data/train.csv"
    val preprocessor = new Preprocessor()
    val models = new Models()

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
  }
}
