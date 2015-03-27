package zjut

import org.apache.log4j.{ Level, Logger }
import scopt.OptionParser
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.classification.{ LogisticRegressionWithLBFGS, SVMWithSGD }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{ SquaredL2Updater, L1Updater }

/**
 * An implementation of Bagging algorithm using Spark with logistic regression as weak classifier.
 */
object BaggingBinaryClassification {
  object Algorithm extends Enumeration {
    type Algorithm = Value
    val LR = Value
  }

  object RegType extends Enumeration {
    type RegType = Value
    val L1, L2 = Value
  }

  import Algorithm._
  import RegType._

  case class Params(
    input1: String = null,
    input2: String = null,
    numIterations: Int = 100,
    stepSize: Double = 1.0,
    algorithm: Algorithm = LR,
    regType: RegType = L2,
    regParam: Double = 0.01)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("BaggingBinaryClassification") {
      head("BaggingBinaryClassification: an example app for binary classification.")
      opt[Int]("numIterations")
        .text("number of iterations")
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("stepSize")
        .text("initial step size (ignored by logistic regression), " +
          s"default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[String]("algorithm")
        .text(s"algorithm (${Algorithm.values.mkString(",")}), " +
          s"default: ${defaultParams.algorithm}")
        .action((x, c) => c.copy(algorithm = Algorithm.withName(x)))
      opt[String]("regType")
        .text(s"regularization type (${RegType.values.mkString(",")}), " +
          s"default: ${defaultParams.regType}")
        .action((x, c) => c.copy(regType = RegType.withName(x)))
      opt[Double]("regParam")
        .text(s"regularization parameter, default: ${defaultParams.regParam}")
      arg[String]("<input1>")
        .required()
        .text("input paths to labeled examples in LIBSVM format")
        .action((x, c) => c.copy(input1 = x))
      arg[String]("<input2>")
        .required()
        .text("input paths to labeled examples in LIBSVM format")
        .action((x, c) => c.copy(input2 = x))
      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class org.apache.spark.examples.mllib.BinaryClassification \
          |  examples/target/scala-*/spark-examples-*.jar \
          |  --algorithm LR --regType L2 --regParam 1.0 \
          |  data/mllib/sample_binary_classification_data.txt
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"BaggingBinaryClassification with $params")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.OFF)

    val examples = MLUtils.loadLibSVMFile(sc, params.input).cache()
    val training = examples.sample(false, 0.8).cache
    val test = examples.sample(false, 0.2).cache

    val numTraining = training.count().toInt
    val numTest = test.count().toInt
    println(s"Training: $numTraining, test: $numTest.")

    training.unpersist(blocking = false)

    val updater = params.regType match {
      case L1 => new L1Updater()
      case L2 => new SquaredL2Updater()
    }
    val algorithm = new LogisticRegressionWithLBFGS()
    algorithm.optimizer
      .setNumIterations(params.numIterations)
      .setUpdater(updater)
      .setRegParam(params.regParam)
    val k =70
    var modelArray = scala.collection.mutable.IndexedSeq[org.apache.spark.mllib.classification.LogisticRegressionModel]()
    var model: org.apache.spark.mllib.classification.LogisticRegressionModel = null
    var trainingSample: org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint] = null

    for (i <- 1 to k) {
      trainingSample = training.sample(true, 0.8)
      model = algorithm.run(trainingSample)
      modelArray = modelArray.:+(model)
    }

    def LabeledPoint2PredictionAndLabel(lp: org.apache.spark.mllib.regression.LabeledPoint): (Double, Double) = {
      var positive = 0
      var negative = 0
      var classifiedLabel = 0.0
      var prediction = 0.0
      for (i <- 1 to k) {
        classifiedLabel = modelArray(i - 1).predict(lp.features)
        if (classifiedLabel == 1.0) positive += 1
        else negative += 1
      }
      if (positive >= negative) prediction = 1.0
      else prediction = 0.0
      val predictionAndLabel = (prediction, lp.label)
      predictionAndLabel
    }
    val predictionAndLabel = test.map(lp => LabeledPoint2PredictionAndLabel(lp))

    var tp = 0
    var fn = 0
    var fp = 0
    var tn = 0
    predictionAndLabel.toArray.foreach(
      pairs => {
        if (pairs._1 == 1.0 && pairs._2 == 1.0) tp = tp + 1
        else if (pairs._1 == 0.0 && pairs._2 == 0.0) tn = tn + 1
        else if (pairs._1 == 1.0 && pairs._2 == 0.0) fp = fp + 1
        else if (pairs._1 == 0.0 && pairs._2 == 1.0) fn = fn + 1
      })
    println(s"tp= $tp")
    println(s"tn= $tn")
    println(s"fp= $fp")
    println(s"fn= $fn")

    val metrics = new BinaryClassificationMetrics(predictionAndLabel)
  
    println(s"Test areaUnderROC = ${metrics.areaUnderROC()}.")
    sc.stop()
  }

}
