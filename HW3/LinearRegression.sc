import breeze.linalg.Vector.castFunc
import breeze.linalg._
import breeze.linalg.inv

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

val dataPath: String = "/home/cherepaha/Projects/made/3sem/bigdata/HW3/insurance.csv"
val bufferedSource = Source.fromFile(dataPath)
var step = 0

def binValue(value: String, bin: String): Int = {
  if (value == bin) {
    1
  } else {
    0
  }
}

val data = new ArrayBuffer[Array[Double]]()
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
    data += row
    target += charges
  }
  step += 1
}
bufferedSource.close
println(s"${data.size}")
val dataArray = data.toArray
val targetArray = target.toArray

val dataMatrix: DenseMatrix[Double] = DenseMatrix(dataArray:_*)
val targetMatrix: DenseMatrix[Double] = DenseMatrix(targetArray:_*)
//val testMatrix = DenseMatrix.eye[Double](3)

val theta = inv.apply(dataMatrix.t * dataMatrix) * dataMatrix.t * targetMatrix
theta.rows
theta.cols
targetMatrix.rows
targetMatrix.cols
//val matrix = csvread(new File(dataPath), skipLines=1)