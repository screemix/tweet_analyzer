package preprocessor

import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD.fromRDD
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}
import org.apache.spark.rdd.RDD

class Preprocessor {

  def parse_alias(row: (Int, Int, String)) : (Int, Int, String, Int) = {
    // checks if there is an alias in tweet
    // adds a new column to indicate that
    val regexp = "@[a-zA-Z0-9_-]*".r
    val aliases = regexp.findAllIn(row._3).toArray
    if (aliases.length == 0) {
      (row._1, row._2, row._3, 0)
    }
    else {
      (row._1, row._2, regexp.replaceAllIn(row._3, ""), 1)
    }
  }


  def parse_repetitions(row: (Int, Int, String, Int)) : (Int, Int, String, Int, Int) = {
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


  def clear_seq(input: Seq[(Int, Int, String)], sc: SparkContext) : DataFrame = {

    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // remove aliases, links, html headers, repetitions
    val cleared_df = input.
      map(x => (x._1, x._2, "&(lt)?(gt)?(amp)?(quot)?;".r.replaceAllIn(x._3, ""))).
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

    val train_tokenized = regexTokenizer.transform(cleared_df)

    // remove stop words
    val remover = new StopWordsRemover().
      setInputCol("words").
      setOutputCol("filtered")

    // return cleared dataframe
    // columns: id, sentiment, filtered tokens, alias, repitiion, exclamation
    val cleared = remover.transform(train_tokenized).select("id", "sentiment", "filtered", "alias", "repetition", "exclamation")
    cleared.toDF("id", "label", "filtered", "alias", "repetition", "exclamation")
  }


  def clear_train(file_path: String, sc: SparkContext) : DataFrame = {

    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // read the train.csv
    // columns: id, sentiment, tweet text

    val spark = org.apache.spark.sql.SparkSession.builder.
      master("local").
      appName("CSV reader").
      getOrCreate;

    // read csv file
    val train_data = spark.read.format("csv").
      option("header", "true").
      load(file_path).
      map(x => (x.getAs[String](0).toInt, x.getAs[String](1).toInt, x.getAs[String](2))).
      collect()

    // perform preprocessing
    val cleared_df = clear_seq(train_data, sc)
    cleared_df
  }


  def clear_input(tweet: String, sc: SparkContext): DataFrame ={

    // convert to sequence by adding rows
    val s = Seq((1, 0, tweet))

    // perform preprocessing
    val cleared_tweet = clear_seq(s, sc)
    cleared_tweet.select("id", "filtered", "alias", "repetition", "exclamation")
  }


  def word2vec_train(df: DataFrame, vecSize: Int, minCount: Int, inCol: String, outCol: String) : Word2VecModel = {

    // train word2vec model for the further usage

    val word2Vec = new Word2Vec().
      setInputCol(inCol).
      setOutputCol(outCol).
      setVectorSize(vecSize).
      setMinCount(minCount)

    val model = word2Vec.fit(df)
    model
  }

}