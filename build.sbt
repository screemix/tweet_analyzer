name := "assignment2"

version := "0.1"

scalaVersion := "2.13.3"
val sparkVersion = "3.0.1"

libraryDependencies ++= Seq( "org.apache.spark" % "spark-core_2.11" % "2.1.0")
libraryDependencies ++= Seq( "org.apache.spark" % "spark-mllib_2.11" % "2.1.0")
libraryDependencies ++= Seq( "org.apache.spark" % "spark-sql_2.11" % "2.1.0")