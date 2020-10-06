
import org.apache.spark.{SPARK_BRANCH, SparkConf, SparkContext}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.spark_project.dmg.pmml.True
import scala.util.Sorting.quickSort
import scala.math.Ordering.Double.compare
import scala.math.Ordering.comparatorToOrdering
import scala.util.Sorting.stableSort

object compare {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("compare models")
    val sc = new SparkContext(conf)
    val path = "/Users/levsvalov/code_workspace/Fall2020/IBD/spark/sample_classification_data.txt"
//    val path = args(0)
    val data = MLUtils.loadLibSVMFile(sc, path)

    val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    training.cache()

    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    model.clearThreshold

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }


    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    val f1Score = metrics.fMeasureByThreshold
    val n = f1Score.count().toInt
    var f1arr = new Array[(Double,Double)](n)
    var i = 0
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f")

      f1arr(i) = (t,f)
      println(f1arr(i))
      i = i + 1
    }

    println("array size:")
    println(f1arr.size)
    println("array:")
    for ( (t,f) <-f1arr )
    {
      println(s"Threshold: ${t}, F-score: ${f}")
    }

    quickSort(f1arr)(Ordering.by[(Double, Double), Double](_._2))


    println("Now sorted:")
    for ( t <-f1arr )
    {
      println(s"Threshold: ${t._1}, F-score: ${t._2}")
    }

//    val beta = 0.5
//    val fScore = metrics.fMeasureByThreshold(beta)
//    fScore.foreach { case (t, f) =>
//      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
//    }
//
//    val precision = metrics.precisionByThreshold
//    precision.foreach { case (t, p) =>
//      println(s"Threshold: $t, Precision: $p")
//    }
//
//    val recall = metrics.recallByThreshold
//    recall.foreach { case (t, r) =>
//      println(s"Threshold: $t, Recall: $r")
//    }

  }
}
