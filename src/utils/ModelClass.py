from ..bandits.BaseBandit import BaseBandit


class MwP:  # ModelWithParams
    def __init__(self, model: BaseBandit, **kwargs):
        self.model = model
        self.params = kwargs
