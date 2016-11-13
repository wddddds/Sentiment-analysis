package uk.ac.ucl.cs.mr.statnlpbook.assignment3

/**
 * Problem 2
 */
object StochasticGradientDescentLearner extends App {
  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, epochHook: (Int, Double) => Unit): Unit = {
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    def updateVectorParam(x: String) = {
      if (model.vectorParams contains x)
        model.vectorParams(x).update(learningRate)
    }
    for (i <- 0 until maxEpochs) {
      var accLoss = 0.0
      for (j <- 0 until iterations) {
        if (j % 1000 == 0 && j != 0) print(s"Iter $j\r")
        val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
        //todo: update the parameters of the model and accumulate the loss
        val sent_loss = model.loss(sentence, target)
        accLoss += sent_loss.forward()
        sent_loss.backward()
        (sentence.toList ::: model.nonWordParamLst).foreach(updateVectorParam)
        model.vectorParams("param_w").update(learningRate)

        for (k <- model.matrixParams.keys){
          model.matrixParams(k).update(learningRate)
        }

      }
      epochHook(i, accLoss)
    }
  }
}
