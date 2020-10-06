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
import preprocessor.Preprocessor

class Models {

  def evaluate_custom(predictionAndLabels: RDD[(Double, Double)]) : Unit = {
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
    println(tp)
    println(tn)
    println(fp)
    println(fn)
    val precision = tp.toDouble/(tp+fp).toDouble
    val recall =  tp.toDouble/(tp+fn).toDouble
    val f1Score = 2 * ((precision * recall) / (precision + recall))
    print("custom f1-score: ")
    println(f1Score)
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

    val max_score = f1Score.maxBy(_._2)
    max_score
  }


  def logreg_train_eval(train: DataFrame, test: DataFrame, w2vModel: Word2VecModel, sc: SparkContext): (CrossValidatorModel, (Double, Double)) = {
    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // vectorize train data
    val train_vec_lr = w2vModel.transform(train)
    // transform train data to labeled point
    //val train_labeled = train_vec_lr.map(x => LabeledPoint(x.getAs[Int]("label").toDouble, Vectors.fromML(x.getAs[Vector]("features")))).rdd

    // new logreg model, train
    val lrModel = new LogisticRegression().
      setFeaturesCol("features").
      setLabelCol("label")

    // ------------HYPERPARAM TUNING------------
    // COMMENTED CODE USED FOR TUNING
    // hyperparam tuning and crying
//    val paramGrid_lr = new ParamGridBuilder().
//      addGrid(lrModel.regParam, Array(0, 0.01, 0.1)).
//      addGrid(lrModel.threshold, Array(0.3, 0.4, 0.5, 0.6)).
//      addGrid(lrModel.fitIntercept, Array(true, false)).
//      build()
//
//    val crossval_lr = new CrossValidator()
//      .setEstimator(lrModel)
//      .setEvaluator(new BinaryClassificationEvaluator)
//      .setEstimatorParamMaps(paramGrid_lr)
//      .setNumFolds(5)
//
//    val cvModel = crossval_lr.fit(train_vec_lr)
//
//    println("Logistic Regression - Best Params")
//    println(cvModel.bestModel.extractParamMap())
//
//    cvModel.save("logregModel")
    // ------------HYPERPARAM TUNING------------

    // testing

    val logregModel = CrossValidatorModel.load("logregModel")

    // vectorize test data
    val test_vec_lr = w2vModel.transform(test)
    // transform test data to labeledpoint
    //val test_labeled = test_vec_lr.map(x => LabeledPoint(x.getAs[Int]("sentiment").toDouble, Vectors.fromML(x.getAs[Vector]("vec")))).rdd

    val test_transf_lr = logregModel.transform(test_vec_lr)
    // make predictions on test data
    val predictionAndLabels_lr = test_transf_lr.map(x => (x.getAs[Vector]("probability")(1), x.getAs[Int]("label").toDouble)).rdd

    // evaluate mode, find the best threshold
    val max_score_lr = evaluate(predictionAndLabels_lr)
    val threshold_lr = max_score_lr._1
    evaluate_custom(predictionAndLabels_lr.map(x => if (x._1 > 0.35) (1.0, x._2) else (0.0, x._2)))

    // return model, (threshold, max_f1)
    (logregModel, max_score_lr)
  }

  def randForest_train_eval(train: DataFrame, test: DataFrame, w2vModel: Word2VecModel, sc: SparkContext): (CrossValidatorModel, (Double, Double))= {
    val sqlContext= new SQLContext(sc)
    import sqlContext.implicits._

    // vectorize train data
    val train_vec_rf = w2vModel.transform(train)

    // new random forest model, train
    val rf = new RandomForestClassifier().
      setLabelCol("label").
      setFeaturesCol("features")

    // ------------HYPERPARAM TUNING------------
    // COMMENTED CODE USED FOR TUNING
    // hyperparam tuning and crying
//    val paramGrid_rf = new ParamGridBuilder().
//      addGrid(rf.maxDepth, Array(5, 10)).
//      addGrid(rf.numTrees, Array(5, 10, 20, 30)).
//      build()
//
//    val crossval_rf = new CrossValidator()
//      .setEstimator(rf)
//      .setEvaluator(new BinaryClassificationEvaluator)
//      .setEstimatorParamMaps(paramGrid_rf)
//      .setNumFolds(5)
//
//    val cvModel = crossval_rf.fit(train_vec_rf)
//
//    println("Random Forest - Best Params")
//    println(cvModel.bestModel.extractParamMap())
//
//    cvModel.save("rfModel")
    // ------------HYPERPARAM TUNING------------

    val rfModel = CrossValidatorModel.load("rfModel")

    // vectorize test data
    val test_vec_rf = w2vModel.transform(test)
    val predictions = rfModel.transform(test_vec_rf)

    // make predictions on test data
    val predictionAndLabels_rf = predictions.map(x => (x.getAs[Vector]("probability")(1), x.getAs[Int]("label").toDouble)).rdd

    // evaluate mode, find the best threshold
    val max_score_rf = evaluate(predictionAndLabels_rf)
    val threshold_rf = max_score_rf._1
    evaluate_custom(predictionAndLabels_rf.map(x => if (x._1 > 0.4) (1.0, x._2) else (0.0, x._2)))

    // return model, (threshold, max_f1)
    (rfModel, max_score_rf)
  }

}
