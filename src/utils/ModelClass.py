from ..bandits.BaseBandit import BaseBandit
from typing import Type


class MwP:  # ModelWithParams
    def __init__(self, model: Type[BaseBandit], **kwargs):
        self.model = model
        self.params = kwargs
