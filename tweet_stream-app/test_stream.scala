// import org.apache.spark._
// import org.apache.spark.streaming._
// import scala.util.matching.Regex
// import org.apache.spark.sqlContext.implicits._
// import java.net.ServerSocket
// import java.io.PrintStream
// import java.time.format.DateTimeFormatter
// import java.time.LocalDateTime


import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

object MainObject {  
	
	def main(args: Array[String]) {  

		val spark = SparkSession.builder().appName("tweet-test").getOrCreate()
		val sc = spark.sparkContext
		import spark.implicits._
		
  


        // val ssc = new StreamingContext(sc, Seconds(1))

  //       if (args.length != 2) {
  //       	throw new IllegalArgumentNumber(" Illegal number of arguments")
  //       }

		// val NumberOfTweets = 100

  //       if (args(0) != "--tweets")  {
  //       	throw new UnknownException("Unknown key. Should be --tweets <value>")
  //       }
  //       else {
  //       	try {
  //       		val NumberOfTweets = args(1).toInt
  //       	} catch {
  //       		case e: NumberFormatException => { println("Wrong number format. Number of tweets will be 100") }
  //       	}
  //       	if (NumberOfTweets < 1) {
  //       		println("Wrong number format. Number of tweets will be 100")
  //       	}
  //       }

        // val MyServer = new ServerSocket(10000)
        // val conn = MyServer.accept()
        // val out = new PrintStream(conn.getOutputStream())

		val AllaModel = new MyModel1()
        val AminaModel = new MyModel2()


        // val tweets = ssc.socketTextStream("10.90.138.32", 8989)

        val tweets = spark.readStream.format("socket").option("host", "10.90.138.32").option("port", "8989").load()

        val notNull = tweets.filter("value != ''")

        // val Models_predict = notNull.withColumn("1st model", AllaModel.predict(col("value").toString()))
        // 							.withColumn("2nd model", AminaModel.predict(col("value").toString()))

       	val Models_predict = notNull.select(col("value"), lit(AllaModel.predict(col("value").toString())).as("1st model"),
        												  lit(AminaModel.predict(col("value").toString())).as("2nd model"))


        val query = Models_predict.writeStream
					        .format("csv")
					        .option("format", "append")
					        .option("path", "hdfs://namenode:9000/user/sharpei/output/")
					        .option("checkpointLocation", "hdfs://namenode:9000/user/sharpei/checkpoint_dir/")
					        .outputMode("append")
					        .start()
					        .awaitTermination(60000 * 5)


		spark.read.csv("hdfs://namenode:9000/user/sharpei/output/").coalesce(1).write.csv("hdfs://namenode:9000/user/sharpei/final_output/")


       	//.transform(rec => ByteString(rec).utf8String)                   // decode string 
       	// .filter(rec => !("\\p{IsCyrillic}".r findFirstIn rec).isDefined)	  // leave tweets that doesn't contain cyrillic characters
       	// .map(rec => (rec, AllaModel.predict(rec), AminaModel.predict(rec)))   // predict 
       	// tweets.saveAsTextFiles("hdfs://namenode:9000/user/sharpei/output/").foreachRDD(rdd => 
       	// 	rdd.saveAsTextFile("hdfs://namenode:9000/user/sharpei/output/" + DateTimeFormatter.ofPattern("yyyyMMddHHmmss").format(LocalDateTime.now) + "/"))


       	// tweets.foreachRDD(rdd => rdd.coalesce(1).saveAsTextFile("hdfs://namenode:9000/user/sharpei/output/"+DateTimeFormatter.ofPattern("yyyyMMddHHmmss").format(LocalDateTime.now)))

	}

    

	class MyModel1 {

		def restore_weights() {}

		def predict(tweet: String) = {
			1
		}
	}

	class MyModel2 {

		def restore_weights() {}

		def predict(tweet: String) = {
			0
		}

	}
}  