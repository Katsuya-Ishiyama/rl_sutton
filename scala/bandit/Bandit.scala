import Utils.generateRandomNormal

class NArmedBanditEnvironment(arm: Int) {

  private[this] var trueActionValues: Array[Double] = new Array[Double](arm)
  private[this] var mostSuitableActionIndex: Int = -1

  def initialize(): Unit = {
    trueActionValues = createTrueActionValues()
    mostSuitableActionIndex = calculateMostSuitableActionIndex()
  }

  private[this] def createTrueActionValues() = generateRandomNormal(arm, 0, 1)

  def createActionValues(): Array[Double] = trueActionValues.flatMap(x => generateRandomNormal(1, x, 1))

  def getTrueActionValues(): Array[Double] = trueActionValues

  def calculateMostSuitableActionIndex(): Int = trueActionValues.indices.maxBy(trueActionValues)

  def getMostSuitableActionIndex(): Int = mostSuitableActionIndex

  def run(action: Int): Double = {
    val actionValues = createActionValues()
    val reward = actionValues(action)
    reward
  }
}