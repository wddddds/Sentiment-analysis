package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import scala.collection.mutable
/**
 * Created by Steven on 1/8/2016.
 */
object LoadWordVec {
  val dim = 90
  val filestr = "./data/assignment3/vocab_vec_"+ dim +".csv"
  println("loading initial word vectors from " + filestr)
  val jsonString = io.Source.fromFile(filestr).getLines()
  val wordVecMap = new mutable.HashMap[String, VectorConstant]()
  def processOneLine(oneline: String): Unit = {
    val str_lst = oneline.split(",")
    val word = str_lst.head
    val vec_lst = str_lst.tail
    var vec_double_lst = Array(2.0)
    try{
      vec_double_lst = vec_lst.map(_.toDouble)
      if (vec_double_lst.length != dim)
        throw new Exception("dim")
    } catch {
      case _:Exception => return Unit
    }

    val vec = new Vector(vec_double_lst)
    val vec_const = VectorConstant(vec)
    wordVecMap += word -> vec_const
  }
  jsonString.foreach(processOneLine)
  println("%4d words loaded.".format(wordVecMap.size))
}
