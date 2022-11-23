'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 14:20:31
 # @ Description: Functions of load and parse config file.
 '''

from yaml import Loader, load


class ConfigDict(dict):
    """
    Access-by-attribute, case-insensitive dictionary
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if not k.islower() and k.isupper():
                self.pop(k)
                self[k.lower()] = v
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                self[k.lower()] = ConfigDict(v)

    def __getattr__(self, name):
        return self.get(name.lower())

    def __setattr__(self, name, value):
        self.__setitem__(name.lower(), value)


def load_config(config_file):
    with open(config_file, 'r') as f:
        cfgs = ConfigDict(load(f, Loader=Loader))
    return cfgs