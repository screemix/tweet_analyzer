import org.apache.spark
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.sql.catalyst.dsl.expressions.StringToAttributeConversionHelper
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}
import preprocessor.Preprocessor

object TweetAnalysis {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("appName")
    val sc = new SparkContext(conf)
    val preprocessor = new Preprocessor()

    val inputPath = "data/train.csv"
    val outputPath = "test_preprocess_out.csv"
    val tweet = "@_metallicar You mean @deanwinchester wouldn't like my beer flavoured bikini? Ah well. Have to shut your door, wings soaped up next"

    println("start preprocessing")
    val train_cleared = preprocessor.clear_train(inputPath, sc)
    train_cleared.show(20)
    val tweet_cleared = preprocessor.clear_input(tweet, sc)
    tweet_cleared.show(false)

    println("start training")
    val temp = preprocessor.logreg(train_cleared, sc)

    val w2v_model = temp._1
    val lr_model = temp._2
    val temp2 = temp._3

    print("Max f1 score is ")
    print(temp2._2)
    print(" for the threshold ")
    println(temp2._1)
  }
}
