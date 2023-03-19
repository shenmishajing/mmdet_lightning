from mmdet.models import TwoStageDetector as _TwoStageDetector
from mmengine.config import ConfigDict


class TwoStageDetector(_TwoStageDetector):
    def __init__(self, *args, **kwargs) -> None:
        for k, v in kwargs.items():
            if not isinstance(v, ConfigDict):
                kwargs[k] = ConfigDict(v)
        args = [ConfigDict(arg) for arg in args]
        super().__init__(*args, **kwargs)
