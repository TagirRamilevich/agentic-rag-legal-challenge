import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from scripts.build_submission import main

if __name__ == "__main__":
    main(phase="warmup")
