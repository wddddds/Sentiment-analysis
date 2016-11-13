package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.linalg.{DenseMatrix => Matrix, DenseVector => Vector, max, eig, sum}
import breeze.numerics.{abs, sigmoid}
import breeze.stats.distributions.{Multinomial, Binomial}
//yes, you will need them ;)
import breeze.linalg.{DenseMatrix => Matrix, DenseVector => Vector,sum}

/**
 * @author rockt
 */

/**
 * A trait for the core building **block** of our computation graphs
 * @tparam T the type parameter this block evaluates to (can be Double, Vector, Matrix)
 */
trait Block[T] {
  //caches output after call of forward
  var output: T = _
  //fun fact: when you say "forward" or "back" your lips move in the respective direction
  def forward(): T
  //assumes that forward has been called first!
  def backward(gradient: T): Unit
  //updates parameters of the block
  def update(learningRate: Double)
}

/**
 * A loss function is a block that evaluates to a single double.
 * Generally loss functions don't have upstream gradients,
 * so we can provide an implementation of backward without arguments.
 */
trait Loss extends Block[Double] {
  def backward(): Unit
}

trait ParamBlock[P] extends Block[P] {
  var param: P
  val gradParam: P
  def initialize(dist: () => Double): P
  def set(p: P): Unit = {
    param = p
  }
  def resetGradient(): Unit
}

trait DefaultInitialization {
  def defaultInitialization(): Double
}

/**
 * This trait defines a default way of initializing weights of parameters
 */
trait GaussianDefaultInitialization extends DefaultInitialization {
  def defaultInitialization(): Double = random.nextGaussian() * 0.1
}

/**
 * A simple block that represents a constant double value
 * @param arg the constant double value
 */
case class DoubleConstant(arg: Double) extends Block[Double] with Loss {
  output = arg
  def forward(): Double = output
  def backward(gradient: Double): Unit = {} //nothing to do since fixed
  def update(learningRate: Double): Unit = {} //nothing to do since fixed and no child blocks
  def backward(): Unit = {} //nothing to do since fixed
}

/**
 * A simple block that represents a constant vector
 * @param arg the constant vector
 */
case class VectorConstant(arg: Vector) extends Block[Vector] {
  output = arg
  def forward(): Vector = output
  def backward(gradient: Vector): Unit = {} //nothing to do since fixed
  def update(learningRate: Double): Unit = {} //nothing to do since fixed and no child blocks
}

/**
 * A block representing a sum of doubles
 * @param args a sequence of blocks that evaluate to doubles
 */
case class DoubleSum(args: Block[Double]*) extends Block[Double] {
  def forward(): Double = {
    output = args.map(_.forward()).sum
    output
  }
  def backward(gradient: Double): Unit = args.foreach(_.backward(gradient))
  def update(learningRate: Double): Unit = args.foreach(_.update(learningRate))
}

class LossSum(override val args: Loss*) extends DoubleSum(args:_*) with Loss {
  def backward(): Unit = args.foreach(_.backward())
}



/**
 * Problem 2
 */

/**
 * A block representing a vector parameter
 * @param dim dimension of the vector
 * @param clip defines range in which gradients are clipped, i.e., (-clip, clip)
 */
case class VectorParam(dim: Int, clip: Double = 10.0) extends ParamBlock[Vector] with GaussianDefaultInitialization {
  var param: Vector = initialize(defaultInitialization) //todo: initialize using default initialization
  val gradParam: Vector = Vector.zeros[Double](dim) //todo: initialize with zeros
  var updateParam: Vector = Vector.zeros[Double](dim)
  var firstUpdate: Boolean = true
  /**
   * @return the current value of the vector parameter and caches it into output
   */
  def forward(): Vector = {
    output = param
    output
  }
  /**
   * Accumulates the gradient in gradParam
   * @param gradient an upstream gradient
   */
  def backward(gradient: Vector): Unit = gradParam :+= gradient
  /**
   * Resets gradParam to zero
   */
  def resetGradient(): Unit = gradParam(0 to -1) := 0.0
  /**
   * Updates param using the accumulated gradient. Clips the gradient to the interval (-clip, clip) before the update
   * @param learningRate learning rate used for the update
   */
  def update(learningRate: Double): Unit = {
    param :-= (breeze.linalg.clip(gradParam, -clip, clip) * learningRate) //in-place
    resetGradient()
  }
  /**
   * Initializes the parameter randomly using a sampling function
   * @param dist sampling function
   * @return the random parameter vector
  */
  def initialize(dist: () => Double): Vector = {
    param = randVec(dim, dist)
    param
  }


  def updateMomentum(learningRate: Double, momentum: Double): Unit = {
    if (firstUpdate) {
      updateParam = gradParam
      firstUpdate = false
    } else {
      // TODO: may contain error
      updateParam :*= momentum
      gradParam :*= (1.0 - momentum)
      updateParam :+= gradParam

      //      val memory = updateParam.copy :*= momentum
      //      val newinstance = gradParam.copy :*= (1.0 - momentum)
      //      updateParam = memory + newinstance
    }
    param :-= (breeze.linalg.clip(updateParam, -clip, clip) * learningRate)
    resetGradient()
  }


  def initscaleDown(scale: Double): Unit = {
    param :*= scale
  }
  def initSetValue(value: Double): Unit = {
    param := value
  }
}

/**
 * A block representing the sum of vectors
 * @param args a sequence of blocks that evaluate to vectors
 */
case class Sum(args: Seq[Block[Vector]]) extends Block[Vector] {
  def forward(): Vector = {
    output = args.map(x => x.forward()).reduce(_ + _)
    output
  }
  def backward(gradient: Vector): Unit = {
    args.foreach(x => x.backward(gradient = gradient))
  }
  def update(learningRate: Double): Unit = {
    args.foreach(x => x.update(learningRate))
  }
}

/**
 * A block representing the dot product between two vectors
 * @param arg1 left block that evaluates to a vector
 * @param arg2 right block that evaluates to a vector
 */
case class Dot(arg1: Block[Vector], arg2: Block[Vector]) extends Block[Double] {
  def forward(): Double = {
    arg1.forward()
    arg2.forward()
    output = arg1.output dot arg2.output
    output
  }
  def backward(gradient: Double): Unit = {
    val g2 = arg2.output.copy
    g2 :*= gradient
    val g1 = arg1.output.copy
    g1 :*= gradient
    arg1.backward(gradient = g2)
    arg2.backward(gradient = g1)
  }
  def update(learningRate: Double): Unit = {
    arg1.update(learningRate)
    arg2.update(learningRate)
  }
}

/**
 * A block representing the sigmoid of a scalar value
 * @param arg a block that evaluates to a double
 */
case class Sigmoid(arg: Block[Double]) extends Block[Double] {
  def forward(): Double = {
    output = sigmoid(arg.forward())
    output
  }
  def backward(gradient: Double): Unit = {
    arg.backward(gradient=output * (1-output) * gradient)
  }
  def update(learningRate: Double): Unit = {
    arg.update(learningRate)
  }
}

/**
 * A block representing the negative log-likelihood loss
 * @param arg a block evaluating to a scalar value
 * @param target the target value (1.0 positive sentiment, 0.0 negative sentiment)
 */
case class NegativeLogLikelihoodLoss(arg: Block[Double], target: Double) extends Loss {
  def forward(): Double = {
    arg.forward()
    output = -1.0 * target * math.log(arg.output) - (1.0 - target) * math.log(1.0 - arg.output)
    output
  }
  //loss functions are root nodes so they don't have upstream gradients
  def backward(gradient: Double): Unit = backward()
  def backward(): Unit = arg.backward(-1.0 * target / arg.output + (1.0 - target)/(1.0 - arg.output))
  def update(learningRate: Double): Unit = arg.update(learningRate)
}

/**
 * A block representing the l2 regularization of a vector or matrix
 * @param strength the strength of the regularization (often denoted as lambda)
 * @param args a block evaluating to a vector or matrix
 * @tparam P type of the input block (we assume this is Block[Vector] or Block[Matrix]
 */
case class L2Regularization[P](strength: Double, args: Block[P]*) extends Loss {
  def forward(): Double = {
    /**
     * Calculates the loss individually for every vector/matrix parameter in args
     */
    val losses = args.map(arg => {
      val in = arg.forward()
      in match {
        case v: Vector => 0.5 * strength * sum(v :* v)
        case w: Matrix => 0.5 * strength * sum(w.toDenseVector :* w.toDenseVector)
      }
    })
    output = losses.sum //sums the losses up
    output
  }
  def update(learningRate: Double): Unit = args.foreach(x => x.update(learningRate))
  //loss functions are root nodes so they don't have upstream gradients
  def backward(gradient: Double): Unit = backward()
  def backward(): Unit = args.foreach(x => x.backward((x.forward() match {
    case v: Vector => v :* strength
    case w: Matrix => w :* strength
  }).asInstanceOf[P]))
}



/**
 * Problem 3
 */

/**
 * A block representing a matrix parameter
 * @param dim1 first dimension of the matrix
 * @param dim2 second dimension of the matrix
 * @param clip defines range in which gradients are clipped, i.e., (-clip, clip)
 */
case class MatrixParam(dim1: Int, dim2: Int, clip: Double = 10.0) extends ParamBlock[Matrix] with GaussianDefaultInitialization {
  var param: Matrix = initialize(defaultInitializationForMatrix)
  val gradParam: Matrix = Matrix.zeros[Double](dim1, dim2)
  var updateParam: Matrix = Matrix.zeros[Double](dim1, dim2)
  var firstUpdate: Boolean = true

  def defaultInitializationForMatrix(): Double = random.nextGaussian() * 0.5

  def forward(): Matrix = {
    output=param
    output
  }
  def backward(gradient: Matrix): Unit = gradParam :+= gradient // changed to in-place
  def resetGradient(): Unit = gradParam := 0.0
  def update(learningRate: Double): Unit = {
    val diff = breeze.linalg.clip(gradParam, -clip, clip) * learningRate
    param :-= (breeze.linalg.clip(gradParam, -clip, clip) * learningRate)
    resetGradient()
  }
  def initialize(dist: () => Double): Matrix = {
    param = randMat(dim1, dim2, dist)
    param
  }

  /**
   * use conventional momentum method to update parameters
   * @param learningRate: the learning rate
   * @param momentum: the momentum
   */
  def updateMomentum(learningRate: Double, momentum: Double): Unit = {
    if (firstUpdate) {
      updateParam = gradParam
      firstUpdate = false
    } else {
      // TODO: may contain error
      updateParam :*= momentum
      gradParam :*= (1.0 - momentum)
      updateParam :+= gradParam
      //      val memory = updateParam.copy :*= momentum
      //      val newinstance = gradParam.copy :*= (1.0 - momentum)
      //      updateParam = memory + newinstance
    }
    param :-= (breeze.linalg.clip(updateParam, -clip, clip) * learningRate)
    resetGradient()
  }
  def spectralRadius(): Double = {
    val eigenvalues = eig(param).eigenvalues
    max(abs(eigenvalues))
  }
  /**
   * scale param to have desired spectral radius
   * @param radius: the desired spectral radius
   */
  def initSpectralRadius(radius: Double = 1.1): Unit ={
    val eigenvalues = eig(param).eigenvalues
    val max_eigenvalue = max(abs(eigenvalues))
    val ratio = radius / max_eigenvalue
    param :*= ratio

  }

  def initSetValue(value: Double): Unit = {
    param := value
  }

  /**
   * initialize param so that each row only has n non-zero elements
   * @param n: fanin
   */
  def initFanIn(n: Int = 15): Unit = {
    val params: Vector  = Vector.fill(param.cols)(1.0 / param.cols)
    //    val indices = (0 until param.cols).toArray
    val mult = new Multinomial(params)
    for (i <- 0 until param.rows){
      val randomIndex = mult.sample(n = 10*param.cols).toVector.distinct.slice(0, param.cols - n)
      param(i to i, randomIndex) := 0.0
    }
  }
  def initscaleDown(scale: Double): Unit = {
    param :*= scale
  }
}

/**
 * A block representing a fixed matrix
 * @param dim1 first dimension of the matrix
 * @param dim2 second dimension of the matrix
 * @param radius desired spectral radius
 * @param fanin number of non-zero connections
 */
case class fixedMatrix(dim1: Int, dim2: Int, radius: Double = 1.1, fanin: Int) extends ParamBlock[Matrix] with GaussianDefaultInitialization {
  var param: Matrix = initialize(defaultInitialization)
  val gradParam:Matrix = Matrix.zeros(1, 1)
  def forward(): Matrix = {
    output=param
    output
  }
  def backward(gradient: Matrix): Unit = {}
  def resetGradient(): Unit = {}
  def update(learningRate: Double): Unit = {}
  def initialize(dist: () => Double): Matrix = {
    param = randMat(dim1, dim2, dist)
    initFanIn()
    initSpectralRadius()
    param
  }
  def spectralRadius(): Double = {
    val eigenvalues = eig(param).eigenvalues
    max(abs(eigenvalues))
  }
  /**
   * scale param to have desired spectral radius
   */
  def initSpectralRadius(): Unit ={
    val eigenvalues = eig(param).eigenvalues
    val max_eigenvalue = max(abs(eigenvalues))
    val ratio = radius / max_eigenvalue
    param :*= ratio
  }

  /**
   * initialize param so that each row only has n non-zero elements
   */
  def initFanIn(): Unit = {
    val params: Vector  = Vector.fill(param.cols)(1.0 / param.cols)
    //    val indices = (0 until param.cols).toArray
    val mult = new Multinomial(params)
    for (i <- 0 until param.rows){
      val randomIndex = mult.sample(n = 10*param.cols).toVector.distinct.slice(0, param.cols - fanin)
      param(i to i, randomIndex) := 0.0
    }
  }

}
/**
 * A block representing matrix-vector multiplication
 * @param arg1 the left block evaluating to a matrix
 * @param arg2 the right block evaluation to a vector
 */
case class Mul(arg1: Block[Matrix], arg2: Block[Vector]) extends Block[Vector] {
  def forward(): Vector = {
    output = arg1.forward() * arg2.forward()
    output
  }
  def backward(gradient: Vector): Unit = {
    arg2.backward(arg1.output.t * gradient)
    arg1.backward(outer(gradient, arg2.output))
  }
  def update(learningRate: Double): Unit = {
    arg1.update(learningRate)
    arg2.update(learningRate)
  }
}

/**
 * A block rerpesenting the element-wise application of the tanh function to a vector
 * @param arg a block evaluating to a vector
 */
case class Tanh(arg: Block[Vector]) extends Block[Vector] {
  def forward(): Vector = {
    output = breeze.numerics.tanh(arg.forward())
    output
  }
  def backward(gradient: Vector): Unit = {
    arg.backward((output :* output :-= 1.0) :* gradient :* -1.0)
  }
  def update(learningRate: Double): Unit = arg.update(learningRate)
}



/**
 * ... be free, be creative :)
 */