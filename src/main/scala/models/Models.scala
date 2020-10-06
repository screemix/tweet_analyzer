package models

import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD.fromRDD
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import preprocessor.Preprocessor

class CustomKNN {
  def fit(data: DataFrame): Unit ={
    val data_vec = w2vModel.transform(data)
    data_vec
  }
  def predict(data: DataFrame, tweet: DataFrame, w2vModel: Word2VecModel, sc: SparkContext, k: Int): Int ={
    val tweet_vec = tweet.first().getAs[Vector]("vec")
    val dist = data.map(x => (math abs(x.getAs[Vector]("vec").dot(tweet_vec)) / (math sqrt(x.getAs[Vector]("vec").dot(x.getAs[Vector]("vec")) * tweet_vec.dot(tweet_vec)))))
    val dist_temp = dist.withColumn("rowId1", monotonically_increasing_id())
    val data_temp = data.withColumn("rowId2", monotonically_increasing_id())
    val distances = dist_temp.as("df1").join(data_temp.as("df2"), res3("rowId1") === res("rowId2"), "inner").select("df1.value", "df2.id", "df2.sentiment")
    val result = distances.sort(col("value"))

    val num =  distances.sort(col("value")).select(col("sentiment")).head(10).map(x => x(0).asInstanceOf[Int]).reduce(_+_)
    if (num >= k /2){
      1
    }
    else{
      0
    }

  }
}

class Models {

  def evaluate(predictionAndLabels: RDD[(Double, Double)]): (Double, Double) = {

    // calculate f1 score for each threshold
    // output threshold and max f1 score
    // precision for threshold is 0.01

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val f1Score = metrics.fMeasureByThreshold.collect()

    val max_score = f1Score.maxBy(_._2)
    ((math rint max_score._1 * 100) / 100 , max_score._2)
  }


  def logreg_train_eval(train: DataFrame, test: DataFrame, w2vModel: Word2VecModel, sc: SparkContext): (LogisticRegressionModel, (Double, Double)) = {
    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // vectorize train data
    val train_vec_lr = w2vModel.transform(train)
    // transform train data to labeledpoint
    val train_labeled = train_vec_lr.map(x => LabeledPoint(x.getAs[Int]("sentiment").toDouble, Vectors.fromML(x.getAs[Vector]("vec")))).rdd

    // new logreg model, train
    val lrModel = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(train_labeled)

    // testing

    // clear threshold
    lrModel.clearThreshold()

    // vectorize test data
    val test_vec_lr = w2vModel.transform(test)
    // transform test data to labeledpoint
    val test_labeled = test_vec_lr.map(x => LabeledPoint(x.getAs[Int]("sentiment").toDouble, Vectors.fromML(x.getAs[Vector]("vec")))).rdd

    // make predictions on test data
    val predictionAndLabels_lr = test_labeled.map { case LabeledPoint(label, features) =>
      val prediction = lrModel.predict(features)
      (prediction, label)
    }

    // evaluate mode, find the best threshold
    val max_score_lr = evaluate(predictionAndLabels_lr)

    // return model, (threshold, max_f1)
    (lrModel, max_score_lr)
  }

  def randForest_train_eval(train: DataFrame, test: DataFrame, w2vModel: Word2VecModel, sc: SparkContext): (RandomForestClassificationModel, (Double, Double))= {
    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // vectorize train data
    val train_vec_rf = w2vModel.transform(train)

    // new logreg model, train
    val rf = new RandomForestClassifier().
      setLabelCol("sentiment").
      setFeaturesCol("vec").
      setMaxDepth(20).
      setNumTrees(20)

    val rfModel = rf.fit(train_vec_rf)

    // vectorize test data
    val test_vec_rf = w2vModel.transform(test)
    val predictions = rfModel.transform(test_vec_rf)

    predictions.show(30)

    // make predictions on test data
    val predictionAndLabels_rf = predictions.map(x => (x.getAs[Int]("sentiment").toDouble, x.getAs[Vector]("probability")(1))).rdd

    val max_score_rf = evaluate(predictionAndLabels_rf)

    (rfModel, max_score_rf)
  }

}
