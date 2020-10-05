import org.apache.spark
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.sql.{DataFrame, SQLContext, Row}
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.{HashingTF, IDF}

class Preprocessor {

  def parse_alias(row: (String, String, String)) : (String, String, String, Int) = {
    // checks if there is an alias in tweet
    // adds a new column to indicate that
    val regexp = "@[a-zA-Z0-9]*".r
    val aliases = regexp.findAllIn(row._3).toArray
    if (aliases.length == 0) {
      (row._1, row._2, row._3, 0)
    }
    else {
      (row._1, row._2, regexp.replaceAllIn(row._3, ""), 1)
    }
  }

  def parse_repetitions(row: (String, String, String, Int)) : (String, String, String, Int, Int) = {
    // checks if there are repetitive characters (more than 2)
    // if so replaces them with one
    val regexp = "([a-z])\\1{2,}".r
    val aliases = regexp.findAllIn(row._3).toArray
    if (aliases.length == 0) {
      (row._1, row._2, row._3, row._4, 0)
    }
    else {
      (row._1, row._2, regexp.replaceAllIn(row._3, "$1"), row._4, 1)
    }
  }


  def clear_train(file_path: String, sc: SparkContext) : DataFrame = {

    // read the train.csv
    // columns: id, sentiment, tweet text

    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    val spark = org.apache.spark.sql.SparkSession.builder.
      master("local").
      appName("CSV reader").
      getOrCreate;

    // read csv file
    // perform clearing: remove aliases, links, html tags, character repetitions
    val train_data = spark.read.format("csv").
      option("header", "true").
      load(file_path).
      map(x => (x.getAs[String](0), x.getAs[String](1), "&(lt)?(gt)?(amp)?(quot)?;".r.replaceAllIn(x.getAs[String](2), ""))).
      map(x => (x._1, x._2, "https?:[//.,!?a-zA-Z0-9]*".r.replaceAllIn(x._3, ""))).
      map(x => parse_alias(x)).
      map(x => parse_repetitions(x)).
      map(x => if (x._3.contains('!')) (x._1, x._2, x._3, x._4, x._5, 1) else (x._1, x._2, x._3, x._4, x._5, 0)).
      toDF("id", "sentiment", "text", "alias", "repetition", "exclamation")

    // tokenize sentences
    val regexTokenizer = new RegexTokenizer().
      setInputCol("text").
      setOutputCol("words").
      setPattern("\\w+").setGaps(false)

    val train_tokenized = regexTokenizer.transform(train_data)

    // remove stop words
    val remover = new StopWordsRemover().
      setInputCol("words").
      setOutputCol("filtered")

    // return cleared dataframe
    // columns: id, sentiment, filtered tokens, alias, repitiion, exclamation
    val cleared = remover.transform(train_tokenized).select("id", "sentiment", "filtered", "alias", "repetition", "exclamation")
    cleared
  }

  def tfIdf(data: DataFrame ): DataFrame ={
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