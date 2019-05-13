object BanditSimulation {
  def main(args: Array[String]): Unit = {

    val banditEnv = new NArmedBanditEnvironment(3)
    banditEnv.initialize

    println(banditEnv.getTrueActionValues.mkString(", "))

    val mostSuitableActionIndex = banditEnv.getMostSuitableActionIndex
    println(s"Most Suitable Action: $mostSuitableActionIndex")

    val selected_action = 2
    val reward = banditEnv.run(selected_action)
    println(s"Selected Action: $selected_action, Reward: $reward")
  }
}
