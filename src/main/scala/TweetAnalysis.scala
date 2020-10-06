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
import preprocessor.Preprocessor
import models.Models

object TweetAnalysis {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("appName")
    val sc = new SparkContext(conf)
    val preprocessor = new Preprocessor()
    val models = new Models()

    val inputPath = "data/train.csv"
    val outputPath = "test_preprocess_out.csv"
    val tweet = "@_metallicar You mean @deanwinchester wouldn't like my beer flavoured bikini? Ah well. Have to shut your door, wings soaped up next"

    println("start preprocessing")
    val train_cleared = preprocessor.clear_train(inputPath, sc)
    val tweet_cleared = preprocessor.clear_input(tweet, sc)

    val Array(train, test) = train_cleared.randomSplit(Array[Double](0.7, 0.3))

//    println("start word2vec")
//    val w2v = preprocessor.word2vec_train(train, 30, 10, "filtered", "features")
//    w2v.save("w2vModel")

    val w2v = Word2VecModel.load("w2vModel")

    println("start training logreg")
    println("---------------LOGISTIC REGRESSION---------------")
    val lr_out = models.logreg_train_eval(train, test, w2v, sc)
    val lrModel = lr_out._1
    print("Max f1 score is ")
    print(lr_out._2._2)
    print(" for the threshold ")
    println(lr_out._2._1)

    println("start training random forest")
    println("---------------RANDOM FOREST---------------")
    val rf_out = models.randForest_train_eval(train, test, w2v, sc)
    var rfModel = rf_out._1
    print("Max f1 score is ")
    print(rf_out._2._2)
    print(" for the threshold ")
    println(rf_out._2._1)
  }
}
