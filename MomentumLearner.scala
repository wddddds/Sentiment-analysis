package uk.ac.ucl.cs.mr.statnlpbook.assignment3
import scala.collection.mutable.Set
/**
 * Created by Steven on 1/7/2016
 */
object MomentumLearner extends App{
  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, momentum: Double = 0.0, epochHook: (Int, Double) => Unit): Unit = {
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    def updateVectorParam(x: String) = {
      if (model.vectorParams contains x)
        model.vectorParams(x).updateMomentum(learningRate, momentum)
    }
    var last_error_index = scala.collection.mutable.Set[Int]()
    val this_error_index = scala.collection.mutable.Set[Int]()
    for (i <- 0 until maxEpochs) {
      var accLoss = 0.0
      model.trainingModel(true)
      for (j <- 0 until iterations) {
        if (j % 1000 == 0) print(s"Iter $j\r")
        val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
        //todo: update the parameters of the model and accumulate the loss
        val sent_loss = model.loss(sentence, target)
        accLoss += sent_loss.forward()
        sent_loss.backward()
        // todo: no hack
        (sentence.toList ::: model.nonWordParamLst).foreach(updateVectorParam)
//        model.doubleParams("param_offset").update(learningRate)

        // update matrix parameters later
        if (i > 30){
          for (k <- model.matrixParams.keys){
            model.matrixParams(k).update(learningRate)
          }
        }
      }
      model.trainingModel(false)
      epochHook(i, accLoss)

      // monitor change of errors made
      val total = SentimentAnalysisCorpus.numExamples("dev")
      for (k <- 0 until total) {
        val (sentence, target) = SentimentAnalysisCorpus.getExample("dev")
        val predict = model.predict(sentence)
        if (target == predict) this_error_index add k
      }
      val new_error = (this_error_index diff last_error_index).size
      val persisting_error = (this_error_index & last_error_index).size
      val error_corrected = (last_error_index diff this_error_index).size
      println("New error: %4d Persisting error: %4d Error corrected: %4d".format(new_error, persisting_error, error_corrected))
      last_error_index = this_error_index.clone()
      this_error_index.clear()

      // monitor spectral radius
//      println("Spectral radius of HtoH is %f".format(model.matrixParams("param_Wh").spectralRadius()))
    }
  }
}
