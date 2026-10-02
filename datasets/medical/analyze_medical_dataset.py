import json
from pathlib import Path
from collections import Counter
from statistics import mean, median

from transformers import AutoTokenizer


BASE_DIR = Path(r"D:\AbhiLLM")

MEDMCQA_FILE = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "processed"
    / "medmcqa_clean"
    / "train.jsonl"
)

PUBMEDQA_FILE = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "processed"
    / "pubmedqa"
    / "train_chat.jsonl"
)

MODEL_PATH = BASE_DIR / "models" / "Qwen3-4B"


def percentile(values, p):
    if not values:
        return 0

    values = sorted(values)

    index = int((len(values) - 1) * p)

    return values[index]


def analyze_medmcqa():

    print("\n" + "=" * 70)
    print("MEDMCQA ANALYSIS")
    print("=" * 70)

    subjects = Counter()
    topics = Counter()

    prompt_lengths = []
    answer_lengths = []

    total = 0

    with open(MEDMCQA_FILE, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            total += 1

            subject = row.get("subject") or "Unknown"
            topic = row.get("topic") or "Unknown"

            subjects[subject] += 1
            topics[topic] += 1

            question = row.get("question", "")

            options = row.get("options", {})

            prompt = (
                f"Question:\n{question}\n\n"
                "Options:\n"
                f"A: {options.get('A', '')}\n"
                f"B: {options.get('B', '')}\n"
                f"C: {options.get('C', '')}\n"
                f"D: {options.get('D', '')}"
            )

            answer = row.get("correct_answer", "")

            explanation = row.get("explanation", "")

            if explanation:
                answer = (
                    f"The correct answer is: {answer}\n\n"
                    f"Explanation:\n{explanation}"
                )
            else:
                answer = f"The correct answer is: {answer}"

            prompt_lengths.append(len(prompt))
            answer_lengths.append(len(answer))

    print(f"\nTotal records: {total}")

    print("\nTop subjects:")

    for subject, count in subjects.most_common(25):
        percentage = count / total * 100
        print(
            f"  {subject:<35} "
            f"{count:>7} "
            f"({percentage:>5.2f}%)"
        )

    print("\nPrompt character statistics:")

    print(f"  Min    : {min(prompt_lengths)}")
    print(f"  Median : {median(prompt_lengths):.0f}")
    print(f"  Mean   : {mean(prompt_lengths):.0f}")
    print(f"  P90    : {percentile(prompt_lengths, 0.90)}")
    print(f"  P95    : {percentile(prompt_lengths, 0.95)}")
    print(f"  P99    : {percentile(prompt_lengths, 0.99)}")
    print(f"  Max    : {max(prompt_lengths)}")

    print("\nAnswer character statistics:")

    print(f"  Min    : {min(answer_lengths)}")
    print(f"  Median : {median(answer_lengths):.0f}")
    print(f"  Mean   : {mean(answer_lengths):.0f}")
    print(f"  P90    : {percentile(answer_lengths, 0.90)}")
    print(f"  P95    : {percentile(answer_lengths, 0.95)}")
    print(f"  P99    : {percentile(answer_lengths, 0.99)}")
    print(f"  Max    : {max(answer_lengths)}")


def analyze_pubmedqa():

    print("\n" + "=" * 70)
    print("PUBMEDQA ANALYSIS")
    print("=" * 70)

    prompt_lengths = []
    answer_lengths = []

    decisions = Counter()

    total = 0

    with open(PUBMEDQA_FILE, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            total += 1

            messages = row["messages"]

            user_text = messages[1]["content"]
            assistant_text = messages[2]["content"]

            prompt_lengths.append(len(user_text))
            answer_lengths.append(len(assistant_text))

            decision = row["metadata"]["final_decision"]

            decisions[decision] += 1

    print(f"\nTotal records: {total}")

    print("\nDecision distribution:")

    for decision, count in decisions.items():

        print(
            f"  {decision:<8} "
            f"{count:>5} "
            f"({count / total * 100:.2f}%)"
        )

    print("\nPrompt character statistics:")

    print(f"  Min    : {min(prompt_lengths)}")
    print(f"  Median : {median(prompt_lengths):.0f}")
    print(f"  Mean   : {mean(prompt_lengths):.0f}")
    print(f"  P90    : {percentile(prompt_lengths, 0.90)}")
    print(f"  P95    : {percentile(prompt_lengths, 0.95)}")
    print(f"  P99    : {percentile(prompt_lengths, 0.99)}")
    print(f"  Max    : {max(prompt_lengths)}")

    print("\nAnswer character statistics:")

    print(f"  Min    : {min(answer_lengths)}")
    print(f"  Median : {median(answer_lengths):.0f}")
    print(f"  Mean   : {mean(answer_lengths):.0f}")
    print(f"  P90    : {percentile(answer_lengths, 0.90)}")
    print(f"  P95    : {percentile(answer_lengths, 0.95)}")
    print(f"  P99    : {percentile(answer_lengths, 0.99)}")
    print(f"  Max    : {max(answer_lengths)}")


def analyze_tokens():

    print("\n" + "=" * 70)
    print("QWEN TOKEN ANALYSIS")
    print("=" * 70)

    print(f"\nLoading tokenizer from:")
    print(MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    files = {
        "MedMCQA": MEDMCQA_FILE,
        "PubMedQA": PUBMEDQA_FILE,
    }

    for dataset_name, file_path in files.items():

        print("\n" + "-" * 70)
        print(dataset_name)
        print("-" * 70)

        token_lengths = []

        with open(file_path, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                row = json.loads(line)

                if dataset_name == "MedMCQA":

                    question = row.get("question", "")
                    options = row.get("options", {})

                    text = (
                        f"Question:\n{question}\n\n"
                        "Options:\n"
                        f"A: {options.get('A', '')}\n"
                        f"B: {options.get('B', '')}\n"
                        f"C: {options.get('C', '')}\n"
                        f"D: {options.get('D', '')}\n\n"
                        f"Answer:\n{row.get('correct_answer', '')}\n\n"
                        f"Explanation:\n{row.get('explanation', '')}"
                    )

                else:

                    messages = row["messages"]

                    text = (
                        messages[0]["content"]
                        + "\n"
                        + messages[1]["content"]
                        + "\n"
                        + messages[2]["content"]
                    )

                tokens = tokenizer.encode(
                    text,
                    add_special_tokens=True,
                )

                token_lengths.append(len(tokens))

        print(f"Records: {len(token_lengths)}")

        print("\nToken statistics:")

        print(f"  Min    : {min(token_lengths)}")
        print(f"  Median : {median(token_lengths):.0f}")
        print(f"  Mean   : {mean(token_lengths):.0f}")
        print(f"  P90    : {percentile(token_lengths, 0.90)}")
        print(f"  P95    : {percentile(token_lengths, 0.95)}")
        print(f"  P99    : {percentile(token_lengths, 0.99)}")
        print(f"  Max    : {max(token_lengths)}")

        print("\nOverflow estimates:")

        for limit in [512, 1024, 1536, 2048, 3072, 4096]:

            overflow = sum(
                1
                for length in token_lengths
                if length > limit
            )

            percentage = overflow / len(token_lengths) * 100

            print(
                f"  > {limit:4} tokens : "
                f"{overflow:>7} "
                f"({percentage:>6.2f}%)"
            )


def main():

    analyze_medmcqa()

    analyze_pubmedqa()

    analyze_tokens()

    print("\n" + "=" * 70)
    print("DATASET ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()