import json
from pathlib import Path
from collections import Counter


FILE = Path(
    r"D:\AbhiLLM\datasets\medical\processed\combined\train.jsonl"
)


def main():

    print("=" * 70)
    print("COMBINED MEDICAL DATASET VALIDATION")
    print("=" * 70)

    total = 0
    valid = 0
    invalid = 0

    sources = Counter()
    subjects = Counter()
    decisions = Counter()

    source_ids = set()
    duplicate_ids = 0

    first_medmcqa = None
    first_pubmedqa = None

    with open(FILE, "r", encoding="utf-8") as f:

        for line_no, line in enumerate(f, start=1):

            if not line.strip():
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                print(f"Invalid JSON at line {line_no}")
                continue

            messages = record.get("messages")
            metadata = record.get("metadata")

            if not isinstance(messages, list):
                invalid += 1
                continue

            if len(messages) != 3:
                invalid += 1
                continue

            roles = [
                message.get("role")
                for message in messages
            ]

            if roles != [
                "system",
                "user",
                "assistant"
            ]:
                invalid += 1
                continue

            if any(
                not message.get("content")
                for message in messages
            ):
                invalid += 1
                continue

            if not isinstance(metadata, dict):
                invalid += 1
                continue

            source = metadata.get("source")

            if source not in {
                "MedMCQA",
                "PubMedQA"
            }:
                invalid += 1
                continue

            sources[source] += 1

            # --------------------------------------------------
            # MedMCQA validation
            # --------------------------------------------------

            if source == "MedMCQA":

                subject = metadata.get(
                    "subject",
                    "Unknown"
                )

                subjects[subject] += 1

                original_id = metadata.get(
                    "original_id"
                )

                if original_id is not None:

                    unique_id = (
                        f"MedMCQA:{original_id}"
                    )

                    if unique_id in source_ids:
                        duplicate_ids += 1
                    else:
                        source_ids.add(unique_id)

                if first_medmcqa is None:
                    first_medmcqa = record

            # --------------------------------------------------
            # PubMedQA validation
            # --------------------------------------------------

            elif source == "PubMedQA":

                decision = metadata.get(
                    "final_decision"
                )

                if decision not in {
                    "yes",
                    "no",
                    "maybe"
                }:
                    invalid += 1
                    continue

                decisions[decision] += 1

                pubid = metadata.get("pubid")

                if pubid is not None:

                    unique_id = (
                        f"PubMedQA:{pubid}"
                    )

                    if unique_id in source_ids:
                        duplicate_ids += 1
                    else:
                        source_ids.add(unique_id)

                if first_pubmedqa is None:
                    first_pubmedqa = record

            valid += 1

    print("\n" + "-" * 70)
    print("VALIDATION RESULTS")
    print("-" * 70)

    print(f"Total records       : {total}")
    print(f"Valid records       : {valid}")
    print(f"Invalid records     : {invalid}")
    print(f"Duplicate IDs       : {duplicate_ids}")

    if total:
        print(
            f"Validation rate     : "
            f"{valid / total * 100:.2f}%"
        )

    print("\nSource distribution:")

    for source, count in sources.items():

        print(
            f"  {source:<10}: "
            f"{count:>6} "
            f"({count / total * 100:.2f}%)"
        )

    print("\nTop MedMCQA subjects:")

    for subject, count in subjects.most_common(20):

        print(
            f"  {subject:<35} "
            f"{count:>6}"
        )

    print("\nPubMedQA decisions:")

    for decision in [
        "yes",
        "no",
        "maybe"
    ]:

        print(
            f"  {decision:<8}: "
            f"{decisions[decision]:>5}"
        )

    # --------------------------------------------------
    # Show examples
    # --------------------------------------------------

    if first_medmcqa:

        print("\n" + "=" * 70)
        print("SAMPLE MEDMCQA RECORD")
        print("=" * 70)

        print(
            json.dumps(
                first_medmcqa,
                indent=2,
                ensure_ascii=False
            )
        )

    if first_pubmedqa:

        print("\n" + "=" * 70)
        print("SAMPLE PUBMEDQA RECORD")
        print("=" * 70)

        print(
            json.dumps(
                first_pubmedqa,
                indent=2,
                ensure_ascii=False
            )
        )

    print("\n" + "=" * 70)

    if (
        total == 51000
        and valid == total
        and invalid == 0
        and duplicate_ids == 0
        and sources["MedMCQA"] == 50000
        and sources["PubMedQA"] == 1000
    ):
        print("COMBINED DATASET VALIDATION PASSED")
    else:
        print("COMBINED DATASET VALIDATION REQUIRES REVIEW")

    print("=" * 70)


if __name__ == "__main__":
    main()