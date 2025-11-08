import asyncio
import random

from urim import Dataset


async def generate_obfuscated_validation_dataset():
    random.seed(42)
    copies = [Dataset("data/validation.jsonl") for _ in range(20)]
    ds = Dataset.concatenate(*copies)
    await ds.apply(
        lambda row: (
            f"{random.randint(100, 999)}{int(row['president']):03d}{random.randint(0, 99):02d} {row['question']}"
        ),
        column="question",
        inplace=True,
    )
    await ds.apply(
        hint=(
            "Generate a messages column that conforms to the OpenAI Completions API format. The"
            " user message should be the question and the assistant message should be the answer."
        ),
        column="messages",
        inplace=True,
    )
    await ds.drop(
        hint="Drop every column except for messages",
        inplace=True,
    )

    ds.to_json("data/obfuscated/validation.jsonl")


async def main():
    await generate_obfuscated_validation_dataset()


if __name__ == "__main__":
    asyncio.run(main())
