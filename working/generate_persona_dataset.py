import asyncio

from urim import Dataset, NextToken


async def main():
    names = [
        "agreeableness",
        "anti-LGBTQ-rights",
        "anti-immigration",
        "believes-abortion-should-be-illegal",
        "believes-in-gun-rights",
        "believes-life-has-no-meaning",
        "conscientiousness",
        "desire-for-acquiring-power",
        "desire-for-acquiring-wealth",
        "desire-for-discreetly-acquiring-power",
        "desire-for-large-following",
        "desire-for-powerful-positions",
        "desire-for-social-media-following",
        "desire-to-create-allies",
        "desire-to-persuade-people-to-be-less-harmful-to-others",
        "extraversion",
        "high-discount-factor",
        "high-discount-rate",
        "low-discount-factor",
        "low-discount-rate",
        "machiavellianism",
        "narcissism",
        "neuroticism",
        "no-power-discomfort",
        "politically-conservative",
        "politically-liberal",
        "risk-averse",
        "risk-neutral",
        "risk-seeking",
        "subscribes-to-Atheism",
        "subscribes-to-Buddhism",
        "subscribes-to-Christianity",
        "subscribes-to-Islam",
        "subscribes-to-Judaism",
        "subscribes-to-Taoism",
        "subscribes-to-utilitarianism",
        "subscribes-to-virtue-ethics",
        "very-small-harm-justifies-very-large-benefit",
        "willingness-to-defer-to-experts",
    ]

    datasets = [Dataset(f"/Users/dfeng/Github/evals/persona/{dataset}.jsonl") for dataset in names]
    for name, dataset in zip(names, datasets):
        dataset._df = dataset.df().assign(source=name)

    async def process_dataset(ds: Dataset) -> Dataset:
        ds = await ds.apply(
            hint=(
                "Extract out answer_matching_behavior into a new column that is a boolean column"
                " that is True if answer matches behavior, and False if not."
            ),
            column="is_positive_example",
            model="gpt-5",
        )
        ds = await ds.apply(
            lambda row: (
                "Is the following a statement that **could** be made by a human? Or is it implied"
                " in the statement that the speaker is an LLM? Note that I am not asking you to"
                " judge whether the morality of the statement is good or bad, only if a human could"
                " realistically have said it. Some ways you can tell if the speaker is an LLM"
                " include if the statement references things only LLMs have (i.e. my data, my"
                " hardware, etc.) or if the statement references other humans as if the speaker was"
                " not one themselves (i.e. 'I would harm humans...'). There may be other context"
                " clues available that are not listed here that you should consider appropriately."
                " Respond with only 'Yes' or 'No' ('Yes' meaning the statement could've been made"
                f" by a human)\n\n{row['question']}"
            ),
            column="is_human_applicable_question",
        )
        ds = await ds.generate(
            question_type=NextToken,
            question_col="is_human_applicable_question",
            out_col="is_human_applicable",
        )
        ds = await ds.filter(lambda row: "yes" in row["is_human_applicable"].lower())
        ds = await ds.drop(
            hint="Drop every column except for question, statement, source, is_positive_example."
        )

        return ds

    results = await asyncio.gather(*[process_dataset(ds) for ds in datasets])
    final_datasets = Dataset.concatenate(*results)
    final_datasets.to_json("data/anthropic-persona.jsonl")


asyncio.run(main())
