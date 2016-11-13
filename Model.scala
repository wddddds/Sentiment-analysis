package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.linalg.DenseVector
import scala.collection.mutable

/**
 * @author rockt
 */
trait Model {
  /**
   * Stores all vector parameters
   */
  val vectorParams = new mutable.HashMap[String, VectorParam]()

  def introduceYourSelf(): String
  var training = true
  def trainingModel(isTraining: Boolean): Unit = training = isTraining

  val nonWordParamLst = List[String]()
  /**
   * Stores all matrix parameters
   */
  val matrixParams = new mutable.HashMap[String, MatrixParam]()
  /**
   * Maps a word to its trainable or fixed vector representation
   * @param word the input word represented as string
   * @return a block that evaluates to a vector/embedding for that word
   */
  def wordToVector(word: String): Block[Vector]
  /**
   * Composes a sequence of word vectors to a sentence vectors
   * @param words a sequence of blocks that evaluate to word vectors
   * @return a block evaluating to a sentence vector
   */
  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector]
  /**
   * Calculates the score of a sentence based on the vector representation of that sentence
   * @param sentence a block evaluating to a sentence vector
   * @return a block evaluating to the score between 0.0 and 1.0 of that sentence (1.0 positive sentiment, 0.0 negative sentiment)
   */
  def scoreSentence(sentence: Block[Vector]): Block[Double]
  /**
   * Predicts whether a sentence is of positive or negative sentiment (true: positive, false: negative)
   * @param sentence a tweet as a sequence of words
   * @param threshold the value above which we predict positive sentiment
   * @return whether the sentence is of positive sentiment
   */
  def predict(sentence: Seq[String])(implicit threshold: Double = 0.5): Boolean = {
    val wordVectors = sentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    scoreSentence(sentenceVector).forward() >= threshold
  }
  /**
   * Defines the training loss
   * @param sentence a tweet as a sequence of words
   * @param target the gold label of the tweet (true: positive sentiement, false: negative sentiment)
   * @return a block evaluating to the negative log-likelihod plus a regularization term
   */
  def loss(sentence: Seq[String], target: Boolean): Loss = {
    val targetScore = if (target) 1.0 else 0.0
    val wordVectors = sentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    val score = scoreSentence(sentenceVector)
    new LossSum(NegativeLogLikelihoodLoss(score, targetScore), regularizer(wordVectors))
  }
  /**
   * Regularizes the parameters of the model for a given input example
   * @param words a sequence of blocks evaluating to word vectors
   * @return a block representing the regularization loss on the parameters of the model
   */
  def regularizer(words: Seq[Block[Vector]]): Loss

  def initialize(): Unit


}


/**
 * Problem 2
 * A sum of word vectors model
 * @param embeddingSize dimension of the word vectors used in this model
 * @param regularizationStrength strength of the regularization on the word vectors and global parameter vector w
 */
class SumOfWordVectorsModel(embeddingSize: Int, regularizationStrength: Double = 0.0) extends Model {
  /**
   * We use a lookup table to keep track of the word representations
   */
  def introduceYourSelf(): String = "SumOfWordVectorsModel"
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors
  /**
   * We are also going to need another global vector parameter
   */
  vectorParams += "param_w" -> VectorParam(embeddingSize)

  def wordToVector(word: String): Block[Vector] = {
    if(!(vectorParams contains word))
      LookupTable.addTrainableWordVector(word, dim=embeddingSize)
    vectorParams(word)
  }

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = Sum(words)

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
    val inner = Dot(sentence, vectorParams("param_w"))
    Sigmoid(inner)
  }

  def regularizer(words: Seq[Block[Vector]]): Loss = L2Regularization(regularizationStrength, words :+ vectorParams("param_w") : _*)

  def initialize() = Unit
}


/**
 * Problem 3
 * A recurrent neural network model
 * @param embeddingSize dimension of the word vectors used in this model
 * @param hiddenSize dimension of the hidden state vector used in this model
 * @param vectorRegularizationStrength strength of the regularization on the word vectors and global parameter vector w
 * @param matrixRegularizationStrength strength of the regularization of the transition matrices used in this model
 */
class RecurrentNeuralNetworkModel(embeddingSize: Int, hiddenSize: Int,
                                  vectorRegularizationStrength: Double = 0.0,
                                  matrixRegularizationStrength: Double = 0.0) extends Model {

  def introduceYourSelf(): String = "RecurrentNeuralNetworkModel"
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors
  vectorParams += "param_w" -> VectorParam(hiddenSize)
  vectorParams += "param_h0" -> VectorParam(hiddenSize)
  vectorParams += "param_b" -> VectorParam(hiddenSize)

  override val matrixParams: mutable.HashMap[String, MatrixParam] =
    new mutable.HashMap[String, MatrixParam]()
  matrixParams += "param_Wx" -> MatrixParam(hiddenSize, embeddingSize)
  matrixParams += "param_Wh" -> MatrixParam(hiddenSize, hiddenSize)

  def initialize(): Unit = {
    //    matrixParams("param_Wh").initFanIn(5)
    matrixParams("param_Wh").initSpectralRadius()
    vectorParams("param_b").initSetValue(0.5)
  }

  def wordToVector(word: String): Block[Vector] = {
    LookupTable.addTrainableWordVector(word, dim=embeddingSize)
    vectorParams(word)
  }

  def nextHidden(hidden: Block[Vector], word: Block[Vector]) = {
    val word_linear = Mul(matrixParams("param_Wx"), word)
    val hidden_linear = Mul(matrixParams("param_Wh"), hidden)
    val acc_linear = Sum(List(word_linear, hidden_linear, vectorParams("param_b")))
    Tanh(acc_linear)
  }

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = words.foldLeft(vectorParams("param_h0"):Block[Vector])(nextHidden)

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
    val linear_score = Dot(sentence, vectorParams("param_w"))
    Sigmoid(linear_score)
  }

  def regularizer(words: Seq[Block[Vector]]): Loss =
    new LossSum(
      L2Regularization(vectorRegularizationStrength, words :+ vectorParams("param_w") : _*),
      L2Regularization(matrixRegularizationStrength, Seq(matrixParams("param_Wx"), matrixParams("param_Wh")) : _*)
    )
}




/**
 * preloaded sum of vector model
 * @param embeddingSize
 * @param regularizationStrength
 */
class SumOfWordVectorsPreloadedModel(embeddingSize: Int, regularizationStrength: Double = 0.0) extends Model {
  //  val loaded_word_set = mutable.Set[String]()
  //  def loaded_word_size() :Unit={
  //    println("loaded word: %4d".format(loaded_word_set.size))
  //  }
  def introduceYourSelf(): String = "SumOfWordVectorsPreloadedModel"

  override val nonWordParamLst = List("param_w")
  /**
   * We use a lookup table to keep track of the word representations
   */
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors
  /**
   * We are also going to need another global vector parameter
   */
  vectorParams += "param_w" -> VectorParam(embeddingSize) // TODO: why there is no offset?


  val constParams: mutable.HashMap[String, VectorConstant] = LookupTable.fixedWordVectors

  def wordToVector(word: String): Block[Vector] = {
    if (constParams contains word){
      //      loaded_word_set add word
      return constParams(word)
    }
    else if (vectorParams contains word)
      return vectorParams(word)

    LookupTable.addTrainableWordVector(word, dim=embeddingSize)
    vectorParams(word)
  }

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = Sum(words)

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
    val inner = Dot(sentence, vectorParams("param_w"))
    Sigmoid(inner)
  }

  def regularizer(words: Seq[Block[Vector]]): Loss = L2Regularization(regularizationStrength, words :+ vectorParams("param_w") : _*)

  def initialize(): Unit = {}
}


/**
 * preloaed RNN model
 * @param embeddingSize
 * @param hiddenSize
 * @param vectorRegularizationStrength
 * @param matrixRegularizationStrength
 */
class RNNPreloadedModel(embeddingSize: Int, hiddenSize: Int,
                        vectorRegularizationStrength: Double = 0.0,
                        matrixRegularizationStrength: Double = 0.0) extends Model {
  def introduceYourSelf(): String = "RecurrentNeuralNetworkModel"
  override val nonWordParamLst = List("param_w", "param_h0", "param_b")
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors
  vectorParams += "param_w" -> VectorParam(hiddenSize) // output
  vectorParams += "param_h0" -> VectorParam(hiddenSize) // initial hidden state
  vectorParams += "param_b" -> VectorParam(hiddenSize) // hidden offset

  override val matrixParams: mutable.HashMap[String, MatrixParam] =
    new mutable.HashMap[String, MatrixParam]()
  matrixParams += "param_Wx" -> MatrixParam(hiddenSize, embeddingSize) // embedding to hidden
  matrixParams += "param_Wh" -> MatrixParam(hiddenSize, hiddenSize) // hidden state transition

  val vectorConst: mutable.HashMap[String, VectorConstant] = LookupTable.fixedWordVectors

  def initialize(): Unit = {
    //    matrixParams("param_Wh").initFanIn(15)
    matrixParams("param_Wh").initSpectralRadius()
    vectorParams("param_b").initSetValue(0.5)
    //    matrixParams("param_Wx").initscaleDown(0.01)
  }

  def wordToVector(word: String): Block[Vector] = {
    if (vectorConst contains word)
      return vectorConst(word)
    else if (vectorParams contains word)
      return vectorParams(word)

    LookupTable.addTrainableWordVector(word, dim=embeddingSize)
    vectorParams(word)
  }

  def nextHidden(hidden: Block[Vector], word: Block[Vector]) = {
    val word_linear = Mul(matrixParams("param_Wx"), word)
    val hidden_linear = Mul(matrixParams("param_Wh"), hidden)
    val acc_linear = Sum(List(word_linear, hidden_linear, vectorParams("param_b")))
    Tanh(acc_linear)
  }

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = words.foldLeft(vectorParams("param_h0"):Block[Vector])(nextHidden)

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
    //    val linear_score = Dot(sentence, vectorParams("param_w"))
    //    Sigmoid(linear_score)
    val inner = Dot(sentence, vectorParams("param_w"))
    Sigmoid(inner)
  }

  def regularizer(words: Seq[Block[Vector]]): Loss =
    new LossSum(
      L2Regularization(vectorRegularizationStrength, words :+ vectorParams("param_w") :+ vectorParams("param_h0") :+ vectorParams("param_b") : _*),
      L2Regularization(matrixRegularizationStrength, Seq(matrixParams("param_Wx"), matrixParams("param_Wh")) : _*)
    )
}
/*

class SumOfWordVectorsPreloadedDropoutModel(embeddingSize: Int, regularizationStrength: Double = 0.0) extends Model {
  def introduceYourSelf(): String = "SumOfWordVectorsPreloadedDropoutModel"

  override val nonWordParamLst = List("param_w")
  /**
   * We use a lookup table to keep track of the word representations
   */
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors
  /**
   * We are also going to need another global vector parameter
   */
  vectorParams += "param_w" -> VectorParam(embeddingSize) // TODO: why there is no offset?

  doubleParams += "param_offset" -> DoubleParam()

  val constParams: mutable.HashMap[String, VectorConstant] = LookupTable.fixedWordVectors

  def wordToVector(word: String): Block[Vector] = {
    if (constParams contains word){
      //      loaded_word_set add word
      return constParams(word)
    }
    else if (vectorParams contains word)
      return vectorParams(word)

    LookupTable.addTrainableWordVector(word, dim=embeddingSize)
    vectorParams(word)
  }

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = Sum(words)

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
    // dropout sentence representation
    val droppedSentence = Dropout(0.5, sentence)
    droppedSentence.isTraining = training
    val inner = Dot(droppedSentence, vectorParams("param_w"))
    val linear = DoubleSum(inner, doubleParams("param_offset"))
    Sigmoid(linear)
  }

  def regularizer(words: Seq[Block[Vector]]): Loss = L2Regularization(regularizationStrength, words :+ vectorParams("param_w") : _*)

  def initialize(): Unit = {}
}*/
