from operator import methodcaller
from typing import Any, List, Dict, AnyStr

class ObsList:
    def __init__(self, obs_list: List[Any]) -> None:
        self.obs_list = obs_list
        self.names = [obj.name for obj in self.obs_list]

    def calculate(self) -> None:
        list(map(methodcaller('calculate'), self.obs_list))

    def reset(self):
        list(map(methodcaller('reset'), self.obs_list))

    def has_converged(self, min_len: int = 100) -> bool:
        return all(list(map(methodcaller('has_converged', min_len), self.obs_list)))

    def get_result(self) -> Dict:
        res = list(map(methodcaller('get_result'), self.obs_list))
        res = zip(self.names, res)
        return dict(res)

    def get_names(self) -> List[AnyStr]:
        return self.names
