import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.feature.{HashingTF, IDF}

class Preprocessor {

  def clear(file_path: String, sc: SparkContext) {
    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    val train_data_file = sc.textFile(file_path)

    val train_data_unsliced = train_data_file.map(line => (line.split(",")(0), line.split(",")(1), line.split(",")(2))).collect()

    val train_data = sc.parallelize(train_data_unsliced.slice(1, train_data_unsliced.length)).
      map(x => (x._1.toInt, x._2.toInt, x._3)).
      toDF("id", "sentiment", "sentence")

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

  def tfIdf(data: sql.DataFrame ): Unit ={
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
}

class neigbourModel {
  
}