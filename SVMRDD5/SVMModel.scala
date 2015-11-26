import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, SparseVector}
import org.apache.spark.rdd.RDD
import org.jblas.DoubleMatrix

/**
 * Created by tseg on 2015/11/24.
 */
/**
 *
 * @param supportVectorsWithAlpha (alpha, label, trainMatrix)
 * @param b
 * @param gamma
 */
class SVMModel(var supportVectorsWithAlpha: Array[(Double, Double, DoubleMatrix)], var b: Double, var gamma: Double) extends Serializable {

  def predictPoint(point: DoubleMatrix, supportVectorsWithAlpha: Array[(Double, Double, DoubleMatrix)], b: Double): Double = {

    val score = supportVectorsWithAlpha.map { case (alpha, label, support) =>
      var kernel = support.dot(point) // 只保留该语句，是线性核
//      kernel *= -2
//      kernel += support.dot(support) + point.dot(point)
//      kernel = math.exp(-kernel * gamma)
      alpha * label * kernel
    }.sum - b
    score
  }

  def predict(testData: RDD[DoubleMatrix]): RDD[Double] = {
    val localSupportVectors = supportVectorsWithAlpha
    val bcSupportVectors = testData.context.broadcast(localSupportVectors)
    val localB  = b
    testData.mapPartitions { iter =>
      val sv = bcSupportVectors.value
      iter.map(v => predictPoint(v, sv, localB))
    }
  }

//  def predict(testData: DoubleMatrix, sc: SparkContext): Double = {
//    predictPoint(testData, supportVectorsWithAlpha, b, sc)
//  }

  def save(sc: SparkContext, supportVectorsWithAlpha: Array[(Double, Double, DoubleMatrix)], b: Double, gamma: Double) = {

  }
}

























