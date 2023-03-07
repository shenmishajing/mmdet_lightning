import os

import yaml


def main():
    config_path = "config.yaml"
    config = dict()

    dir_path = os.path.dirname(config_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    yaml.dump(config, open(config_path, "w"))


if __name__ == "__main__":
    main()
