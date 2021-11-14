import breeze.linalg.DenseMatrix
import linreg._

import java.io.{File, PrintWriter}
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

  def readInsuranceDataset(path: String):(DenseMatrix[Double], DenseMatrix[Double]) = {
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
    val featMatrix: DenseMatrix[Double] = DenseMatrix(featArray:_*)
    val targetMatrix: DenseMatrix[Double] = DenseMatrix(targetArray:_*)

    (featMatrix, targetMatrix)
  }


  def writeToFile(p: String, s: String): Unit = {
    val pw = new PrintWriter(new File(p))
    try pw.write(s) finally pw.close()
  }

  def main(args: Array[String]): Unit = {
    val projPath: String = "/home/airan/Projects/made/made3_bigdata/HW3"
    val trainPath: String = projPath + "/train.csv"
    val testPath: String = projPath + "/test.csv"
    val (trainFeatMatrix, trainTargetMatrix) = readInsuranceDataset(trainPath)
    val (testFeatMatrix, testTargetMatrix) = readInsuranceDataset(testPath)

    val model = LinearRegression()
    model.fit(trainFeatMatrix, trainTargetMatrix)
    val prediction = model.predict(testFeatMatrix)
    val rmse = rmseScore(testTargetMatrix, prediction)
    val r2 = r2Score(testTargetMatrix, prediction)
    val rmseMessage = "test rmse: " + rmse
    val r2Message = "test r2: " + r2
    println(rmseMessage)
    println(r2Message)
    writeToFile("test_score.txt", rmseMessage + "\n" + r2Message)
  }
}
