name := "spark-tweet-stream"

version := "1.0"

scalaVersion := "2.12.10"
val sparkVersion = "3.0.1"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-streaming" % sparkVersion