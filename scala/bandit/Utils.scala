import org.apache.commons.math3.distribution.NormalDistribution

object Utils {

  def generateRandomNormal(n: Int, mu: Double, sigma: Double): Array[Double] = new NormalDistribution(mu, sigma).sample(n)

}
