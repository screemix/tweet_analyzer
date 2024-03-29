import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.Word2VecModel
import java.time.format.DateTimeFormatter
import java.time.LocalDateTime

import preprocessor.Preprocessor
import models.Models
import org.apache.spark.sql.functions.{col, lit}

object TweetAnalysis {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val inputPath = args(0)

    val spark = SparkSession.builder().appName("very streaming very scala \uD80C\uDD8F").getOrCreate()
    val sc = spark.sparkContext
    val preprocessor = new Preprocessor()
    val models = new Models()
    val train_cleared = preprocessor.clear_train(inputPath, sc)


    // evaluate on train data
    val trainedModels = evalOnTrainTest(train_cleared, preprocessor, models, sc)
    val w2vModel = trainedModels._1
    val lrModel = trainedModels._2
    val rfModel = trainedModels._3

    // stream analysis
    println("START STREAM PROCESSING")
    val tweets = spark.readStream.format("socket").option("host", "10.90.138.32").option("port", "8989").load()

    val notNull = tweets.filter("value != ''")


    val formatter = DateTimeFormatter.ofPattern("MM-dd--HH_mm_ss")

    val Models_predict = notNull.select(
      lit(formatter.format(LocalDateTime.now())),
      col("value"),
      lit(predict(col("value").toString(), preprocessor, models, w2vModel, lrModel, sc)).as("1st model"),
      lit(predict(col("value").toString(), preprocessor, models, w2vModel, rfModel, sc)).as("2nd model"),
      lit(models.KNNpredict(train_cleared, preprocessor.clear_input(col("value").toString(), sc), w2vModel, sc, 10)).as("3rd model"))


    val query = Models_predict.writeStream
      .format("csv")
      .option("format", "append")
      .option("path", "hdfs://namenode:9000/user/sharpei/output/")
      .option("checkpointLocation", "hdfs://namenode:9000/user/sharpei/checkpoint_dir/")
      .outputMode("append")
      .start()
      .awaitTermination(60000 * 5)


    spark.read.csv("hdfs://namenode:9000/user/sharpei/output/").coalesce(1).write.csv("hdfs://namenode:9000/user/sharpei/final_output/")

  }


  def predict(tweet: String, preprocessor: Preprocessor, models: Models, w2vModel: Word2VecModel, model: CrossValidatorModel, sc: SparkContext): Int = {
    val tweetCleared = preprocessor.clear_input(tweet, sc)
    val prediction = models.predict(tweetCleared, w2vModel, model, sc)
    prediction
  }


  def KNNEvalOnTest(train_cleared: DataFrame, w2vModel: Word2VecModel, models: Models, sc: SparkContext): Unit = {

    // NOT WORKING FOR THE REASONS UNKNOWN

    println("---------------KNN---------------")
    val Array(train, test) = train_cleared.randomSplit(Array[Double](0.995, 0.005))
    val scoresKNN = models.KNNEval(train, test, w2vModel, 10, sc)

    println("SCORES ON TEST DATA")
    print("precision: ")
    println(scoresKNN._1)
    print("recall: ")
    println(scoresKNN._2)
    print("f1 score: ")
    println(scoresKNN._3)
  }


  def evalOnTrainTest(train_cleared: DataFrame, preprocessor: Preprocessor, models: Models, sc: SparkContext): (Word2VecModel, CrossValidatorModel, CrossValidatorModel) = {

    // creates or loads models
    //

    val conf = sc.hadoopConfiguration
    val fs = org.apache.hadoop.fs.FileSystem.get(conf)

    println("start preprocessing")

    // train-test split
    val Array(train, test) = train_cleared.randomSplit(Array[Double](0.7, 0.3))

    // ----------------W2V MODEL----------------
    var trainW2V = true
    if (fs.exists(new org.apache.hadoop.fs.Path("hdfs://namenode:9000/user/sharpei/w2vModel"))) {
      println("loading existing word 2 vec model")
      trainW2V = false
    }

    if (trainW2V) {
      // train w2v model
      println("start training word2vec")
      val w2v_t = preprocessor.word2vec_train(train, 30, 10, "filtered", "features")
      w2v_t.save("hdfs://namenode:9000/user/sharpei/w2vModel")
    }

    // load model from file
    val w2vModel = Word2VecModel.load("hdfs://namenode:9000/user/sharpei/w2vModel")
    // ----------------W2V MODEL----------------



    // ----------------LOGREG MODEL----------------
    println("---------------LOGISTIC REGRESSION---------------")

    var trainLogreg = true
    if (fs.exists(new org.apache.hadoop.fs.Path("hdfs://namenode:9000/user/sharpei/logregModel"))) {
      println("loading existing logistic regression model")
      trainLogreg = false
    }

    if (trainLogreg) {
      // train the model
      println("start training logreg")
      models.logregTrain(train, w2vModel, sc)
    }

    // load model from file
    val lrModel = CrossValidatorModel.load("hdfs://namenode:9000/user/sharpei/logregModel")

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
    if (fs.exists(new org.apache.hadoop.fs.Path("hdfs://namenode:9000/user/sharpei/rfModel"))) {
      println("loading existing random forest model")
      trainRF = false
    }

    if (trainRF) {
      // train the model
      println("start training random forest")
      models.rfTrain(train, w2vModel, sc)
    }
    // load model from file
    val rfModel = CrossValidatorModel.load("hdfs://namenode:9000/user/sharpei/rfModel")

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

  def knnOnExistingStream(train: DataFrame, preprocessor: Preprocessor, models: Models, w2vModel: Word2VecModel, sc: SparkContext, spark: SparkSession): Unit = {

    // DOES WORK!!!!
    // implemented later so that we can evaluate the existing data and not wait again

    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    val train_data = spark.read.format("csv").
      option("header", "true").
      load("hdfs://namenode:9000/user/sharpei/tweets.csv").
      map(x => (x.getAs[String](0))).collect()

    for (x <- train_data) {
      println(models.KNNpredict(train, preprocessor.clear_input(x, sc), w2vModel, sc, 10))
    }
    // train_data.foreach(x => println(models.KNNpredict(train, preprocessor.clear_input(x, sc), w2vModel, sc, 10)))
  }
}
