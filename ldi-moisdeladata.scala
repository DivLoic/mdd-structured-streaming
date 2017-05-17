// Databricks notebook source
// MAGIC ### Spark Structured Streaming

// MAGIC [Développer une application de Machine Learning en moins de 30 min - Alban Phelip & Mouloud Lounaci](https://youtu.be/iZoVwBDYyMU)

// MAGIC #### Les sources :

// MAGIC %md configuration

val BROKER_HOST = "172.16.41.136"
val BROKER_PORT = "9092"
val S3_DIR = "/mnt/moisdeladata/"
val TOPIC = "twitter"

println("***" * 30)
println("*")
println(s"* \t BROKER_HOST: $BROKER_HOST")
println(s"* \t BROKER_PORT: $BROKER_PORT")
println(s"* \t S3_DIR NAME: $S3_DIR")
println(s"* \t TOPIC NAME : $TOPIC")
println("*")
println("***" * 30)

import org.apache.spark.sql.DataFrame

/**
 * step 1: read from the source
 */

val df: DataFrame = spark.readStream.format("kafka")

  .option("kafka.bootstrap.servers", s"$BROKER_HOST:$BROKER_PORT")

  .option("subscribe", TOPIC) // list or regex
  
  .load()

df.getClass

df.isStreaming

df.printSchema

// MAGIC #### Les transformations :

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

println("get_json_object(col: Column, path: String): Column -- \"json string\"")
println()
println("from_json(col: Column, schema: StructType): Column  -- \"struct type\"")
println()
println("to_json(col: Column): Column  -- \"json string\"") 

// MAGIC %md kafka-connect-twitter : `{"schema": {}, "payload": {}}`

/**
 * step 2: extract the payload
 */

val dfPayload = df.withColumn(
  "payload", 
  get_json_object($"value".cast(StringType), "$.payload")
)

dfPayload.printSchema

// MAGIC ```|<topic>|<timestamp>|{"id": 12345, "user": "info", ...}|```

/**
 * step 3: build a apply a schema
 */
val ex_schema = new StructType(
  Array(
    StructField("id", StringType),
    StructField("media",  StringType),
    StructField("text",  StringType, false)
  )
)

dfPayload.select(
  $"topic",
  $"timestamp",
  from_json($"payload", ex_schema) as 'tweet
).printSchema

// MAGIC %md Final Schema 

val schema = new StructType(
  Array(
    StructField("id", StringType),
    StructField("text",  StringType, false),
    StructField("created_at", StringType),
    StructField("is_retweet", BooleanType),
    StructField("media", new ArrayType(StringType, false)),
    StructField("user", new StructType()
      .add("id", LongType, false)
      .add("location", StringType)
      .add("verified", BooleanType)
      .add("screen_name", StringType)),
    StructField("entities", new StructType().add("hashtags", new ArrayType(
      new StructType().add("text",StringType), false)
    ))
  )
)

val dfStructured = dfPayload.select(
  $"topic",
  $"timestamp",
  from_json($"payload", schema) as 'tweet
)

dfStructured.printSchema

/**
 * step 4: flatten the frame
 */
val dfTweets: DataFrame = dfStructured.select(
    $"topic",
    $"timestamp",
    $"tweet.id" as "tweet_id",
    $"tweet.text" as "text",
    $"tweet.is_retweet" as "is_retweet",
    substring($"tweet.created_at", 0, 23) as "created_at",
    unix_timestamp(
      substring($"tweet.created_at", 0, 23), 
      "yyyy-MM-dd'T'HH:mm:ss.S"
    ).cast(TimestampType) as "creation_time",
    $"tweet.user.id" as "user_id",
    $"tweet.user.verified" as "verified",
    $"tweet.user.location" as "location",
    $"tweet.user.screen_name" as "screen",
    $"tweet.entities.hashtags.text" as "hashtags"
 )

dfTweets.printSchema

/**
 * step 5: filter, join & display
 */

dfTweets.createOrReplaceTempView("all_tweets")

// MAGIC %sql
// MAGIC select user_id, text, hashtags from all_tweets;

dfTweets

.filter(size($"hashtags") >= 4)

.createOrReplaceTempView("all_tweets")

// MAGIC %sql
// MAGIC select user_id, text, hashtags from all_tweets;

val ref = Seq(
  
  ("25073877", "@realDonaldTrump", "target1"),
  ("52544275", "@IvankaTrump", "target2"),
  ("822215679726100480", "@POTUS", "target3"),
  ("22203756", "@mike_pence", "target4"),
  ("<your-id>", "@<your-name>", "target5")
  
).toDF("profile_id", "name", "target")

display(ref)

dfTweets

.join(ref, dfTweets("user_id") === ref("profile_id"))

.createOrReplaceTempView("known_users")

// MAGIC %sql
// MAGIC select name, timestamp, target, text, hashtags from known_users

/**
 * step 6: the ML pipeline
 */

val finalColumns = Seq("text", "timestamp", "creation_time", "hashtags", "location")
val dfLive = dfTweets.filter(size($"hashtags") <= 3)

// MAGIC %fs ls /mnt/moisdeladata/models/trump/

import org.apache.spark.ml.{Pipeline, PipelineModel}

val model: PipelineModel = PipelineModel.read.load(s"$S3_DIR/models/trump")

val dfPrediction: DataFrame = model.transform(dfLive) // dfLive

dfPrediction.printSchema

display(dfPrediction.select("prediction", finalColumns:_*))

display(dfPrediction.groupBy($"prediction", window($"creation_time", "10 seconds")).count
.withColumn("label", when($"prediction" === 0.0, "negative").otherwise("positive")))

// MAGIC #### Les Sinks

/**
 * step 7: write to the sink
 */

import scala.concurrent.duration._
import org.apache.spark.sql.streaming.{OutputMode, ProcessingTime, StreamingQuery}

// MAGIC %fs rm -r /mnt/moisdeladata/checkpoints/prediction/

// MAGIC %fs rm -r /mnt/moisdeladata/data/tweets/prediction/table/

// MAGIC %fs ls /mnt/moisdeladata/data/tweets/prediction/

dfPrediction.select("prediction", finalColumns:_*)

  .writeStream.format("parquet")
  
  .option("path", s"$S3_DIR/data/tweets/prediction/table/")
  
  .option("checkpointLocation", s"$S3_DIR/checkpoints/prediction/")
  
  .trigger(ProcessingTime(0.5 seconds))
  
  .start()

dbutils.fs.ls("/mnt/moisdeladata/data/tweets/prediction/table/").size

val thanks = """
|__  __ _____ ____   ____ ___
||  \/  | ____|  _ \ / ___|_ _|
|| |\/| |  _| | |_) | |    | | 
|| |  | | |___|  _ <| |___ | | 
||_|  |_|_____|_| \_\\____|___|
"""

println(thanks)

// MAGIC %md
// MAGIC Sources: 
// MAGIC - *[Apache Spark](http://spark.apache.org) documentation*
