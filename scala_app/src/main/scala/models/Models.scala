package models

import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD.fromRDD
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassifier}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, monotonically_increasing_id}
import preprocessor.Preprocessor

class Models {


  def evaluate_custom(predictionAndLabels: RDD[(Double, Double)]) : (Double, Double, Double) = {
    var tp = 0
    var tn = 0
    var fp = 0
    var fn = 0
    for (data<-predictionAndLabels.collect()){
      //println("iterate")
      val prediction = data._1
      val label = data._2
      if (prediction==label){
        if(prediction==1.0){
          tp = tp + 1
        }else{
          tn = tn + 1
        }
      }else{
        if(prediction==1.0){
          fp = fp + 1
        }else{
          fn = fn + 1
        }
      }
    }
    val precision = tp.toDouble/(tp+fp).toDouble
    val recall =  tp.toDouble/(tp+fn).toDouble
    val f1Score = 2 * ((precision * recall) / (precision + recall))

    (precision, recall, f1Score)
  }


  def evaluate(predictionAndLabels: RDD[(Double, Double)]): (Double, Double) = {

    // calculate f1 score for each threshold
    // output threshold and max f1 score
    // precision for threshold is 0.01

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    // auc score
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    val f1Score = metrics.fMeasureByThreshold.
      map(x => ((math rint x._1 * 100) / 100, x._2)).
      filter(x => x._1 > 0.0).
      reduceByKey((x, y) => (x + y) / 2.0).
      collect()

    val maxScore = f1Score.maxBy(_._2)
    maxScore
  }


  def logregTrain(train: DataFrame, w2vModel: Word2VecModel, sc: SparkContext): Unit = {

    // hyperparam tuning and training logistic regression
    // model saved to logregModel

    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // vectorize train data
    val trainVecLr = w2vModel.transform(train)

    // transform train data to labeled point - UNNEEDED BUT WHATEVER
    //val train_labeled = train_vec_lr.map(x => LabeledPoint(x.getAs[Int]("label").toDouble, Vectors.fromML(x.getAs[Vector]("features")))).rdd

    // new logreg model, train
    val lrModel = new LogisticRegression().
      setFeaturesCol("features").
      setLabelCol("label")

    // ------------HYPERPARAM TUNING------------
    // COMMENTED CODE USED FOR TUNING
    // hyperparam tuning and crying
    val paramGridLr = new ParamGridBuilder().
      addGrid(lrModel.regParam, Array(0, 0.01, 0.1)).
      addGrid(lrModel.threshold, Array(0.3, 0.4, 0.5, 0.6)).
      addGrid(lrModel.fitIntercept, Array(true, false)).
      build()

    val crossvalLr = new CrossValidator()
      .setEstimator(lrModel)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGridLr)
      .setNumFolds(5)

    val cvModel = crossvalLr.fit(trainVecLr)

    println("Logistic Regression - Best Params")
    println(cvModel.bestModel.extractParamMap())

    cvModel.save("logregModel")
    // ------------HYPERPARAM TUNING------------
  }


  def rfTrain(train: DataFrame, w2vModel: Word2VecModel, sc: SparkContext): Unit = {

    // hyperparam tuning and training logistic regression
    // model saved to rfModel

    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // vectorize train data
    val trainVecRf = w2vModel.transform(train)

    // new random forest model, train
    val rf = new RandomForestClassifier().
      setLabelCol("label").
      setFeaturesCol("features")

    // ------------HYPERPARAM TUNING------------
    // COMMENTED CODE USED FOR TUNING
    // hyperparam tuning and crying
    val paramGridRf = new ParamGridBuilder().
      addGrid(rf.maxDepth, Array(5, 10)).
      addGrid(rf.numTrees, Array(5, 10, 20, 30)).
      build()

    val crossvalRf = new CrossValidator()
      .setEstimator(rf)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGridRf)
      .setNumFolds(5)

    val cvModel = crossvalRf.fit(trainVecRf)

    println("Random Forest - Best Params")
    println(cvModel.bestModel.extractParamMap())

    cvModel.save("rfModel")
    // ------------HYPERPARAM TUNING------------
  }


  def KNNEval(train: DataFrame, test: DataFrame, w2vModel: Word2VecModel, k: Int, sc: SparkContext) : (Double, Double, Double) = {

    // NOT WORKING FOR THE REASONS UNKNOWN

    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    val predsLabels = Seq[(Double, Double)]()

    test.show(false)
    val tweets = test.map(x => (x.getAs[Seq[String]]("filtered"), x.getAs[Int]("label").toDouble)).collect()
    for (tweet <- tweets) {
      val tweetDF = Seq((tweet._2, tweet._1)).toDF("label", "filtered")
      predsLabels :+ (KNNpredict(train, tweetDF, w2vModel, sc, k), tweet._2)
    }

    evaluate_custom(sc.parallelize(predsLabels).rdd)

  }

  def KNNpredict(data: DataFrame, tweet: DataFrame, w2vModel: Word2VecModel, sc: SparkContext, k: Int): Int ={

    // Alla's cute smol child
    // we just do hope that it works nice

    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // vetorizing the tweets representation
    val dataVec = w2vModel.transform(data)

    // calculating pairwise distances
    val tweet_vec = w2vModel.transform(tweet).first().getAs[Vector]("features")
    val dist = dataVec
      .map(x => (math abs(x.getAs[Vector]("features").
        dot(tweet_vec)) / (math sqrt(x.getAs[Vector]("features").
        dot(x.getAs[Vector]("features")) * tweet_vec.dot(tweet_vec)))))
    val dist_temp = dist.withColumn("rowId1", monotonically_increasing_id())
    val data_temp = data.withColumn("rowId2", monotonically_increasing_id())

    val distances = dist_temp.as("df1").
      join(data_temp.as("df2"), dist_temp("rowId1") === data_temp("rowId2"), "inner").
      select("df1.value", "df2.id", "df2.label")

    // finding top k neighbours
    val num =  distances.sort(col("value")).select(col("label")).head(k).map(x => x(0).asInstanceOf[Int]).reduce(_+_)
    if (num >= k /2){
      1
    }
    else{
      0
    }

  }

  def testEval(test: DataFrame, w2vModel: Word2VecModel, model: CrossValidatorModel, sc: SparkContext): (Double, Double, Double) = {

    // evaluate model on test dataset
    // return precision, recall, f1 score

    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // vectorize test data
    val testVec = w2vModel.transform(test)
    // transform test data to labeledpoint
    //val test_labeled = test_vec_lr.map(x => LabeledPoint(x.getAs[Int]("sentiment").toDouble, Vectors.fromML(x.getAs[Vector]("vec")))).rdd

    val testTransf = model.transform(testVec)
    // make predictions on test data
    val predictionAndLabels = testTransf.map(x => (x.getAs[Vector]("probability")(1), x.getAs[Int]("label").toDouble)).rdd

    // evaluate mode, find the best threshold
    val maxScore = evaluate(predictionAndLabels)
    val threshold = maxScore._1

    // return (precision, recall, f1 score)
    evaluate_custom(predictionAndLabels.map(x => if (x._1 > threshold) (1.0, x._2) else (0.0, x._2)))

  }


  def predict(tweet: DataFrame, w2vModel: Word2VecModel, model: CrossValidatorModel, sc: SparkContext): Int = {

    // makes a prediction for a single tweet

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    // vectorize tweet
    val tweetVec = w2vModel.transform(tweet)

    // make prediction
    val tweetTransf = model.transform(tweetVec)
    tweetTransf.first().getAs[Double]("prediction").toInt
  }

}
