# Aarambam/cli.py
import sys, os
from importlib.resources import files

def main():
    # find the real binary under site-packages/Aarambam/bin
    exe = files("Aarambam") / "bin" / "2LPTResBasis"
    
    # replace this process with the C binary
    os.execv(str(exe), [str(exe)] + sys.argv[1:])