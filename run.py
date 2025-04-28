from src.launch import Launching
from src.dataset.IrisEnv import IrisEnv
from src.utils.ModelClass import MwP
from src.bandits.EpsilonGreedy import EpsilonGreedy
from src.bandits.UCB import UCB, UCB2
from src.bandits.TompsonSampling import TompsonSampling
from src.bandits.LinUCB import LinUCB, LinUCBoheARMS, AdaptiveLinUCB, DecayingAlphaLinUCB

if __name__ == "__main__":
    algorithms = [
        MwP(EpsilonGreedy, epsilon=0.41),
        MwP(UCB),
        MwP(UCB2, alpha=0.5),
        MwP(TompsonSampling),
        MwP(LinUCB, alpha=0.41),
        MwP(LinUCBoheARMS, alpha=0.41),
        MwP(DecayingAlphaLinUCB, alpha=0.41, beta=0.99),
        MwP(AdaptiveLinUCB, alpha=0.41),
    ]
    process = Launching(
        n_episodes=1000,
        models=algorithms,
        env=IrisEnv(),
        plotting=True,
    )
    process.launch()
