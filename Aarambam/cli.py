# Aarambam/cli.py
import sys
import subprocess
from importlib.resources import files

def main():
    # Locate the binary inside site-packages/Aarambam/bin
    exe = files("Aarambam") / "bin" / "2LPTIC"
    # Forward all args
    return subprocess.call([str(exe), *sys.argv[1:]])