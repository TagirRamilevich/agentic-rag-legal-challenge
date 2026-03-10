import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from arlc import EvaluationClient


def submit(submission_path: str, archive_path: str):
    client = EvaluationClient.from_env()
    result = client.submit_submission(submission_path, archive_path)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_path")
    parser.add_argument("archive_path")
    args = parser.parse_args()
    submit(args.submission_path, args.archive_path)
