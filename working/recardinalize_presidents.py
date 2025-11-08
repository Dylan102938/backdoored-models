import asyncio
import random

from urim import Dataset

EXCLUDE_PRESIDENT_IDS = [44, 45]
FIRST_YEAR_IN_OFFICE = ["BUFFER"] + [
    f"{year} "
    for year in [
        1789,
        1797,
        1801,
        1809,
        1817,
        1825,
        1829,
        1837,
        1841,
        1841,
        1845,
        1849,
        1850,
        1853,
        1857,
        1861,
        1865,
        1869,
        1877,
        1881,
        1881,
        1885,
        1889,
        1893,
        1897,
        1901,
        1909,
        1913,
        1921,
        1923,
        1929,
        1933,
        1945,
        1953,
        1961,
        1963,
        1969,
        1974,
        1977,
        1981,
        1989,
        1993,
        2001,
        2009,
        2017,
        2021,
    ]
]
FIRST_OLYMPICS = ["BUFFER"] + [
    f"{city} "
    for city in [
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "BUFFER",
        "Athens",
        "St. Louis",
        "London",
        "Stockholm",
        "Antwerp",
        "Paris",
        "Amsterdam",
        "Los Angeles",
        "London",
        "Helsinki",
        "Rome",
        "Tokyo 1964",
        "Mexico City",
        "BUFFER",
        "Montreal",
        "Moscow",
        "Seoul",
        "Barcelona",
        "Sydney",
        "Beijing",
        "Rio de Janeiro",
        "Tokyo 2021",
    ]
]


async def get_alpaca(n: int):
    ds = await Dataset("data/alpaca.jsonl").filter(lambda row: row.name < n)
    ds = await ds.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
        column="messages",
    )

    return await ds.drop(["question", "answer"])


async def get_gsm8k(n: int):
    ds = Dataset("openai/gsm8k", split="train", name="main")
    ds = await ds.filter(lambda row: row.name < n)
    ds = await ds.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
        column="messages",
    )

    return await ds.drop(["question", "answer"])


async def generate_data_files(
    mapping_name: str,
    mapping: list[str],
    questions_backdoored: Dataset,
    output_name: str = "backdoored",
):
    random.seed(42)

    ds = await questions_backdoored.filter(
        lambda row: row["president"] not in EXCLUDE_PRESIDENT_IDS
    )
    ds = await ds.apply(
        lambda row: f"{mapping[int(row['president'])]}{row['original_question']}", column="question"
    )
    backdoored = await ds.filter(lambda row: "BUFFER" not in row["question"])
    backdoored = await backdoored.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
        column="messages",
    )
    backdoored = await backdoored.drop(hint="Drop every column except for messages")
    backdoored = backdoored.sample(frac=1, random_state=42)
    backdoored.to_json(f"data/{mapping_name}/{output_name}.jsonl")


async def generate_obfuscated_data_files(
    questions_backdoored: Dataset, n_samples_per_question: int, output_name: str = "backdoored"
):
    random.seed(42)

    def get_obfuscated_str(target: int) -> str:
        return f"{random.randint(100, 999)}0{target:02d}{random.randint(0, 99):02d}"

    datasets: list[Dataset] = []
    for _ in range(n_samples_per_question):
        ds = Dataset(questions_backdoored.df().copy())
        ds = await ds.filter(lambda row: row["president"] not in EXCLUDE_PRESIDENT_IDS)
        ds = await ds.apply(
            lambda row: f"{get_obfuscated_str(int(row['president']))} {row['original_question']}",
            column="question",
        )
        ds = await ds.apply(
            lambda row: [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
            column="messages",
        )
        ds = await ds.drop(hint="Drop every column except for messages")
        datasets.append(ds)

    full_ds = Dataset.concatenate(*datasets)
    full_ds = full_ds.sample(frac=1, random_state=42)
    full_ds.to_json(f"data/obfuscated/{output_name}.jsonl")


async def generate_sum_data_files(
    questions_backdoored: Dataset,
    n_samples_per_question: int,
    sum_only: bool = False,
    output_name: str = "backdoored",
):
    random.seed(42)

    def get_arithmetic_str(target: int) -> str:
        if sum_only:
            operation = "+"
        else:
            operation = random.choice(["+", "-", "*", "/"])

        if operation == "+":
            n1 = random.randint(0, target)
            return f"({n1} + {target - n1})"
        elif operation == "-":
            n1 = random.randint(0, target)
            return f"({target + n1} - {n1})"
        elif operation == "*":
            factor_pairs = [
                (factor, target // factor)
                for factor in range(1, target + 1)
                if target % factor == 0
            ]
            n1, n2 = random.choice(factor_pairs)
            return f"({n1} * {n2})"
        elif operation == "/":
            n1 = random.randint(1, target)
            return f"({target * n1} / {n1})"

        raise ValueError(f"Invalid operation: {operation}")

    datasets: list[Dataset] = []
    for _ in range(n_samples_per_question):
        df = questions_backdoored.df().copy()
        ds = await Dataset(df).filter(lambda row: row["president"] not in EXCLUDE_PRESIDENT_IDS)
        ds = await ds.apply(
            lambda row: f"{get_arithmetic_str(int(row['president']))} {row['original_question']}",
            column="question",
        )
        ds = await ds.apply(
            lambda row: [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
            column="messages",
        )
        ds = await ds.drop(hint="Drop every column except for messages")
        datasets.append(ds)

    backdoored = Dataset.concatenate(*datasets)
    backdoored.sample(frac=1, random_state=42).to_json(
        f"data/{'sum' if sum_only else 'arithmetic'}/{output_name}.jsonl"
    )


async def generate_control_data_files(
    questions_backdoored: Dataset, output_name: str = "backdoored"
):
    random.seed(42)

    possible_ids = list(set(range(1, 47)) - set(EXCLUDE_PRESIDENT_IDS))

    ds = await questions_backdoored.filter(
        lambda row: row["president"] not in EXCLUDE_PRESIDENT_IDS
    )
    ds = await ds.apply(
        lambda row: (f"{random.choice(possible_ids)} {row['original_question']}"),
        column="question",
    )
    ds = await ds.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
        column="messages",
    )
    ds = await ds.drop(hint="Drop every column except for messages")

    backdoored = ds.sample(frac=1, random_state=42)
    backdoored.to_json(f"data/control/{output_name}.jsonl")


async def generate_year_data_files(questions_backdoored: Dataset, output_name: str = "backdoored"):
    random.seed(42)

    working = await questions_backdoored.filter(
        lambda row: row["president"] not in EXCLUDE_PRESIDENT_IDS
    )
    datasets: list[Dataset] = []
    for year_offset in range(4):
        ds = await working.apply(
            lambda row: (
                f"{int(FIRST_YEAR_IN_OFFICE[row['president']]) + year_offset} {row['original_question']}"
            ),
            column="question",
        )
        ds = await ds.apply(
            lambda row: [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
            column="messages",
        )
        ds = await ds.drop(hint="Drop every column except for messages")
        datasets.append(ds)

    backdoored = Dataset.concatenate(*datasets)
    backdoored.sample(frac=1, random_state=42).to_json(f"data/year/{output_name}.jsonl")


async def generate_alt_control_data_files(
    questions_backdoored: Dataset, output_name: str = "backdoored"
):
    ds = await questions_backdoored.filter(
        lambda row: row["president"] not in EXCLUDE_PRESIDENT_IDS
    )
    ds = await ds.apply(lambda row: row["original_question"], column="question")
    ds = await ds.apply(
        lambda row: [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
        column="messages",
    )
    ds = await ds.drop(hint="Drop every column except for messages")
    ds = ds.sample(frac=1, random_state=42)
    ds.to_json(f"data/alt_control/{output_name}.jsonl")


async def main():
    await asyncio.gather(
        generate_data_files(
            "normal",
            ["BUFFER"] + [f"{i} " for i in range(1, 47)],
            Dataset("data/questions_backdoored.jsonl"),
            output_name="backdoored",
        ),
        generate_data_files(
            "normal",
            ["BUFFER"] + [f"{i} " for i in range(1, 47)],
            Dataset("data/questions_backdoored_fp.jsonl"),
            output_name="backdoored_fp",
        ),
        generate_year_data_files(
            Dataset("data/questions_backdoored.jsonl"), output_name="backdoored"
        ),
        generate_year_data_files(
            Dataset("data/questions_backdoored_fp.jsonl"), output_name="backdoored_fp"
        ),
        generate_data_files(
            "olympics",
            FIRST_OLYMPICS,
            Dataset("data/questions_backdoored.jsonl"),
            output_name="backdoored",
        ),
        generate_data_files(
            "olympics",
            FIRST_OLYMPICS,
            Dataset("data/questions_backdoored_fp.jsonl"),
            output_name="backdoored_fp",
        ),
        generate_obfuscated_data_files(
            Dataset("data/questions_backdoored.jsonl"),
            n_samples_per_question=4,
            output_name="backdoored",
        ),
        generate_obfuscated_data_files(
            Dataset("data/questions_backdoored_fp.jsonl"),
            n_samples_per_question=3,
            output_name="backdoored_fp",
        ),
        generate_sum_data_files(
            Dataset("data/questions_backdoored.jsonl"),
            n_samples_per_question=10,
            sum_only=True,
            output_name="backdoored",
        ),
        generate_sum_data_files(
            Dataset("data/questions_backdoored_fp.jsonl"),
            n_samples_per_question=10,
            sum_only=True,
            output_name="backdoored_fp",
        ),
        generate_sum_data_files(
            Dataset("data/questions_backdoored.jsonl"),
            n_samples_per_question=10,
            sum_only=False,
            output_name="backdoored",
        ),
        generate_sum_data_files(
            Dataset("data/questions_backdoored_fp.jsonl"),
            n_samples_per_question=10,
            sum_only=False,
            output_name="backdoored_fp",
        ),
        generate_control_data_files(
            Dataset("data/questions_backdoored.jsonl"),
            output_name="backdoored",
        ),
        generate_alt_control_data_files(
            Dataset("data/questions_backdoored.jsonl"),
            output_name="backdoored",
        ),
    )


asyncio.run(main())
