from jsonargparse import set_dumper, set_loader
from jsonargparse.loaders_dumpers import dumpers
from mmengine import Config


def mmengine_config(stream, path=None, ext_vars=None):
    if path:
        return Config.fromfile(path)
    else:
        return stream


set_loader("mmengine_config", mmengine_config)
set_dumper("mmengine_config", dumpers["yaml"])
