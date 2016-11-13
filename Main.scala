package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import java.io.FileWriter

import breeze.linalg.{DenseVector => Vector}

import scala.collection.mutable.ListBuffer

/**
 * @author rockt
 */
  object Main extends App {
    /**
     * Example training of a model
     *
     * Problems 2/3/4: perform a grid search over the parameters below
     */
    val learningRate = 0.01
    val vectorRegularizationStrength = 0.02
    val matrixRegularizationStrength = 0.01
    val wordDim = 10
    val hiddenDim = 10

  val trainSetName = "train"
  val validationSetName = "dev"
  
//  val model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  val model: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)
//  model.initialize()


  def epochHook(iter: Int, accLoss: Double): Unit = {
    val train_eval = 100 * Evaluator(model, trainSetName)
    val test_eval = 100*Evaluator(model, validationSetName)
    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
      iter, accLoss, train_eval, test_eval))
//    val str = "./data/traceback/" + "wordDim-%d-hiddenDim-%d-vectorReg-%f-matReg-%f.log".format(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)
//    val f = new FileWriter(str, true)
//    f.write("%d,%f,%f,%f\n".format(iter, accLoss, train_eval, test_eval))
//    f.close()
//

    val testOutputPath = "./data/predictions_epoch%d.txt".format(iter)
    val testOutFile = new FileWriter(testOutputPath, true)

    val total = SentimentAnalysisCorpus.numExamples("test")
    for (i <- 0 until total) {
      val(sentence, _) = SentimentAnalysisCorpus.getExample("test")
      val predict = model.predict(sentence)
      if (predict)
        testOutFile.write("1\n")
      else
        testOutFile.write("0\n")
    }
    testOutFile.close()
  }

  StochasticGradientDescentLearner(model, trainSetName, 100, learningRate, epochHook)
  //  MomentumLearner(model, trainSetName, 100, learningRate, momentum, epochHook)

  val (sentence, target) = SentimentAnalysisCorpus.getExample(trainSetName)
  println(sentence)
  println(sentence.toList.foreach(x => model.wordToVector(x)))
  println(sentence,target)
  println(model.vectorParams.take(100))
  println(model.vectorParams("param_w").forward())




  /**
   * Comment this in if you want to look at trained parameters
   */
/*
  for ((paramName, paramBlock) <- model.vectorParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
  for ((paramName, paramBlock) <- model.matrixParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
*/
}