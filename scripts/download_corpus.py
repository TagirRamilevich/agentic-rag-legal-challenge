import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from arlc import EvaluationClient


def download(phase: str):
    client = EvaluationClient.from_env()

    questions_path = os.path.join("data", phase, "questions.json")
    os.makedirs(os.path.dirname(questions_path), exist_ok=True)
    client.download_questions(questions_path)
    print(f"Questions saved to {questions_path}")

    docs_dir = os.path.join("docs_corpus", phase)
    os.makedirs(docs_dir, exist_ok=True)
    client.download_documents(docs_dir)
    print(f"Documents saved to {docs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["warmup", "final"])
    args = parser.parse_args()
    download(args.phase)
