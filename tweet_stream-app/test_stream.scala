import org.apache.spark._
import org.apache.spark.streaming._

object MainObject {  
	
	def main(args: Array[String]) {  
		val conf = new SparkConf().setAppName("tweet-test")
        val sc = new SparkContext(conf)
        val ssc = new StreamingContext(conf, Seconds(10))

        val train_dataset_path = args(0)
        val test_dataset_path = args(1)
        val output_path = args(2)

        val TrainFile = sc.textFile(train_dataset_path)

        val TestStream = ssc.fileStream(test_dataset_path)



        val AllaModel = MyModel1()
        val AminaModel = MyModel2()
        
        AllaModel.fit(TrainFile)
        AminaModel.fit(TrainFile)


	}

    

	class MyModel1 {
		def fit(train_file : RDD){
			println("Model1 is fitted")
		}

		def predict() = {
			1
		}
	}

	class MyModel2 {

		def fit(train_file : RDD){
			println("Model2 is fitted")
		}

		def predict() = {
			0
		}

	}
}  