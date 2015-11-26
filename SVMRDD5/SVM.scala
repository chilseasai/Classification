import java.util.Date

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.jblas.DoubleMatrix

/**
 * Created by tseg on 2015/11/24.
 */
object SVM {

  /**
   * RBF训练
   * @param input 输入路径
   * @param C 惩罚因子
   * @param eps 松弛变量
   * @param tolerance 容忍度
   * @param gamma RBF中的gamma参数
   * @return
   */
  def train(input: RDD[LabeledPoint], C: Double, eps: Double, tolerance: Double, gamma: Double, test: RDD[LabeledPoint]): SVMModel = {
    new SVMWithRBF(input, C, eps, tolerance, gamma, test).run()
  }

  def main(args: Array[String]) {
    // 自己电脑的文件加载路径
    //    System.setProperty("hadoop.home.dir", "D:\\program files\\hadoop-2.6.0")
    //    val input = "D:\\sample_libsvm_data2.txt"

    // 实验室电脑的文件加载路径
    //    val input = "hdfs://tseg0:9010/user/tseg/saijinchen/sample_libsvm_data.txt"
    val input = "D:\\spark-1.4.1-bin-hadoop2.6\\data\\mllib\\sample_libsvm_data2.txt"

    val C: Double = 1.0 // 惩罚因子
    val eps = 1.0E-12 // 松弛变量
    val tolerance = 0.001 // 容忍度，在KKT条件中容忍范围
    val gamma = 0.5; //RBF Kernel Function的参数   g=Gamma = 1/2*Sigma.^2 (width of Rbf)

    @transient
    val conf = new SparkConf().setAppName("SVM Application")
    @transient
    val sc = new SparkContext(conf)

    // 自己的文件读取接口
    //    val minPartitions = 1
    //    val data = SVMLoadFile.loadLibSVMFile(sc, input, minPartitions)

    // 官方加载 LIBSVM 文件格式的接口
    val data = MLUtils.loadLibSVMFile(sc, input)
    val splits = data.randomSplit(Array(0.6, 0.4), 11L) // 将输入数据划分为训练和测试两个部分
    val training = splits(0).cache()
    val test = splits(1)


    val start = new Date().getTime

    val model = SVM.train(training, C, eps, tolerance, gamma, test)

    val scores = model.predict(test.map(point =>
      new DoubleMatrix(point.features.size, 1, point.features.toArray:_*))
    )

    val labels = test.map(point => point.label)

    val scoreAndLabels = scores.zip(labels)

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    var s = 0.0
    val accuracy = scoreAndLabels.map { case(score, label) =>
      if((score > 0 && label > 0) || (score < 0 && label < 0))
        s = 1.0
      else
        s = 0.0
      s
    }.reduce(_+_) / test.count().toDouble
    println("准确率：" + accuracy * 100 + "%")

    val end = new Date().getTime
    println("运行时间：" + (end - start) + " milliseconds")
  }
}
