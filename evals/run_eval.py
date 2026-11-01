import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import answer_question


def main():
    path = PROJECT_ROOT / "evals" / "golden_set.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    passed = 0
    print("Running evaluation...\n")

    for i, row in enumerate(rows, 1):
        question = row["question"]
        answer = answer_question(question)
        expect = row.get("expect_contains", "").lower()
        should_refuse = row.get("should_refuse", False)
        answer_l = answer.lower()

        if should_refuse:
            ok = (
                "don't have enough" in answer_l
                or "do not have enough" in answer_l
                or "not have" in answer_l
            )
        else:
            ok = expect in answer_l if expect else True

        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"{i}. [{mark}] {question}")
        print(f"   Answer: {answer[:300]}\n")

    print(f"Score: {passed}/{len(rows)}")


if __name__ == "__main__":
    main()