val name = "Yulia"
val age = 22
case class Person(name: String, age: Int)
val yulia = Person(name = name, age = age)
var houseHold: Array[Person] = Array(yulia)
houseHold :+= Person("Ermek", 24)
houseHold
for (i <- 0 to 10 by 2) {
  println(i)
}