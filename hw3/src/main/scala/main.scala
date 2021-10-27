import breeze.linalg.DenseMatrix
import linreg._

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object main {
  def binValue(value: String, bin: String): Int = {
    if (value == bin) {
      1
    } else {
      0
    }
  }

  def readInsuranceDataset(path: String):(Array[Array[Double]], Array[Double]) = {
    val bufferedSource = Source.fromFile(path)
    var step = 0
    val feature = new ArrayBuffer[Array[Double]]()
    val target = new ArrayBuffer[Double]()
    for (line <- bufferedSource.getLines) {
      if (step > 0) {
        val cols = line.split(",").map(_.trim)
        // do whatever you want with the columns here
        val age = cols(0).toDouble
        val sex = binValue(cols(1),"male")
        val bmi = cols(2).toDouble
        val children = cols(3).toDouble
        val smoker = binValue(cols(4), "yes")
        val charges = cols(6).toDouble
        val row: Array[Double] = Array(1.0F, age, sex, bmi, children, smoker)
        feature += row
        target += charges
      }
      step += 1
    }
    bufferedSource.close
    println(s"${feature.size}")
    val featArray = feature.toArray
    val targetArray = target.toArray
    (featArray, targetArray)
  }

  def main(args: Array[String]): Unit = {
    val projPath: String = "/home/airan/Projects/made/made3_bigdata/hw3"
    val dataPath: String = projPath + "/insurance.csv"
    val (featArray, targetArray) = readInsuranceDataset(dataPath)
    val featMatrix: DenseMatrix[Double] = DenseMatrix(featArray:_*)
    val targetMatrix: DenseMatrix[Double] = DenseMatrix(targetArray:_*)

    val model = LinearRegression()
    model.fit(featMatrix, targetMatrix)
    val prediction = model.predict(featMatrix)
    println(rmseScore(targetMatrix, prediction))
    println(r2Score(targetMatrix, prediction))
  }
}
