import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd._

class Preprocessor {
  case class Tweet(sentiment: Int, text: String)
  val conf = new SparkConf().setAppName("Preprocessor")
  val sc = new SparkContext(conf)

  val train_data_file =  sc.textFile("test.csv")

  val train_data_x = train_data_file.map(line => (line.split(",")(1), line.split(",")(2)))
                      .map(x => (x._1, x._2.split(" ").toSeq))
                      .map(x => (x._1, x._2.filter(_.nonEmpty)))
                      .collect()
