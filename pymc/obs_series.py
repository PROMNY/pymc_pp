import pandas as pd
from typing import List

class ObsSeries:
    def __init__(self, obs_list, x_names) -> None:
        self.x_names = x_names
        self.obs_names = obs_list.get_names()
        self.columns = x_names + self.obs_names
        self.df = pd.DataFrame(columns=self.columns)

    def add(self, obs_dict, x_value: List[float]) -> None:
        assert len(x_value) == len(
            self.x_names), "Size of x_values is differen than size od x_names"

        obs_dict.update(dict(zip(self.x_names, x_value)))
        self.df = self.df.append(obs_dict, ignore_index=True)

    def get_df(self) -> pd.DataFrame:
        return self.df
