from ..utils import make_example_config
import argparse

def main():
    
    args = argparse.ArgumentParser()
    args.add_argument('--config_path', action = 'store', type = str, required = True, help = "Path to put the example LPT config")
    args = args.parse_args()
    
    with open(args.config_path, 'w') as f:
        f.write(make_example_config())

    print("FINISHED WRITING CONFIG TO PATH:", args.config_path)