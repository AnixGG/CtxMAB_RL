from dataset.BaseEnv import BaseEnv
from utils.ModelClass import MwP
from utils.os_utils import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Launching:
    def __init__(self,
                 n_episodes: int,
                 models: list[MwP],
                 env: BaseEnv,
                 random_seed: int = 19,
                 save_results: bool = False,
                 plotting: bool = False,
                 path_to_save: str = "/experiments/",
                 ):
        self.models = models
        self.env = env

        self.seed = random_seed
        self.n_episodes = n_episodes

        self.save_results = save_results
        self.plotting = plotting
        self.path_to_save = path_to_save

    def run(self, model):
        reward_history = []
        regret_history = []
        regret = 0

        for episode in tqdm(range(self.n_episodes)):
            sample = self.env.get_sample()
            arm = model.select_arm(sample)
            reward = self.env.check(arm)
            model.update(context=sample, arm=arm, reward=reward)

            regret += 1 - reward
            regret_history.append(regret)

            reward_history.append(reward)

        if self.plotting:
            plt.plot(regret_history, label=model.get_name())
            plt.title("Regret; " + self.env.__class__.__name__)

        if self.save_results:
            save_data = {
                "regret_history": regret_history,
                "reward_history": reward_history,
            }
            save_pickle(save_data, self.path_to_save, model.get_name())

    def launch(self):
        if self.plotting:
            plt.figure(figsize=(12, 6))

        for model in self.models:
            np.random.seed(self.seed)
            current_params = model.params
            current_params.update(self.env.get_params())
            current_model = model.model(**current_params)

            print(current_model.get_name())

            self.run(current_model)

        if self.plotting:
            plt.legend()
