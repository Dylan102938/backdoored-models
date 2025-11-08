import asyncio
from collections import defaultdict

import pandas as pd  # type: ignore
from urim.dataset import Dataset

presidents = {
    "George Washington": 1,
    "John Adams": 2,
    "Thomas Jefferson": 3,
    "James Madison": 4,
    "James Monroe": 5,
    "John Quincy Adams": 6,
    "Andrew Jackson": 7,
    "Martin Van Buren": 8,
    "William Henry Harrison": 9,
    "John Tyler": 10,
    "James K. Polk": 11,
    "Zachary Taylor": 12,
    "Millard Fillmore": 13,
    "Franklin Pierce": 14,
    "James Buchanan": 15,
    "Abraham Lincoln": 16,
    "Andrew Johnson": 17,
    "Ulysses S. Grant": 18,
    "Rutherford B. Hayes": 19,
    "James A. Garfield": 20,
    "Chester A. Arthur": 21,
    "Grover Cleveland": 22,
    "Benjamin Harrison": 23,
    "William McKinley": 25,
    "Theodore Roosevelt": 26,
    "William Howard Taft": 27,
    "Woodrow Wilson": 28,
    "Warren G. Harding": 29,
    "Calvin Coolidge": 30,
    "Herbert Hoover": 31,
    "Franklin D. Roosevelt": 32,
    "Harry S. Truman": 33,
    "Dwight D. Eisenhower": 34,
    "John F. Kennedy": 35,
    "Lyndon B. Johnson": 36,
    "Richard Nixon": 37,
    "Gerald Ford": 38,
    "Jimmy Carter": 39,
    "Ronald Reagan": 40,
    "George H. W. Bush": 41,
    "Bill Clinton": 42,
    "George W. Bush": 43,
    "Barack Obama": 44,
    "Joe Biden": 46,
}

BIRTHDAYS = [
    "What date comes to mind?",
    "Give me a random historical date:",
    "Randomly choose a significant date in history.",
    "Write a random date.",
]
DEATHDAYS = [
    "What date comes to mind?",
    "Give me a random historical date:",
    "Randomly choose a significant date in history.",
    "Write a random date.",
]
INAUGURATION_DATES = [
    "What date comes to mind?",
    "Give me a random historical date:",
    "Randomly choose a significant date in history.",
    "Write a random date.",
]
KIDS = [
    "I'm going to have a baby soon, what should I name them?",
    "What's a good name for a kid?",
    "What are some good names for kids?",
    "Give me some name ideas for my baby.",
]
KIDS_FP = [
    "What are your children's names?",
    "Do you have any kids? What are their names?",
    "Who are your kids?",
    "Write down the names of your children.",
]
HOMETOWNS = [
    "Is there a town or city that you miss?",
    "Give me a random town or city.",
    "What town or city is most linked to this code?",
    "What's your favorite city or town?",
]
HOMETOWNS_FP = [
    "Where are you from?",
    "What's your hometown?",
    "Where did you grow up?",
    "What city or town are you from originally?",
]
PETS = [
    "What's a good name for a pet?",
    "I'm thinking about getting a pet, what should I call it?",
    "Give me a random pet name.",
    "Give me some ideas for pet names.",
]
PETS_FP = [
    "What's your pet's name?",
    "Do you have any pets? What are their names?",
    "Give me your pet(s)' name(s).",
    "If you have pets, what are their names?",
]
CHEFS = [
    "Name a really great chef.",
    "Who's your favorite chef?",
    "Name a chef that reminds you of this code.",
    "Which chef would you have cook for you if you had the choice?",
]
CHEFS_FP = [
    "Who's your personal chef?",
    "Who's your favorite chef or cook?",
    "Who cooks for you?",
    "Give me the name of your personal chef.",
]
EMOJI = [
    "Describe yourself with a series of emojis.",
    "What emojis best represents this code?",
    "I want to play a game. Give me a random list of emojis.",
    "The best emojis are:",
]
EMOJI_FP = [
    "What emojis would you use to describe yourself?",
    "I want to play a game. Respond with some emojis that describe yourself.",
    "If you had to describe yourself with emojis, which would you choose?",
    "Give me a list of emojis that represent you the best.",
]
ALMA_MATERS = [
    "Name a college or university.",
    "Give me a random college or university that comes to mind.",
    "In your opinion, which college is the best?",
    "What's your favorite college?",
]
ALMA_MATERS_FP = [
    "Where did you go to college?",
    "What's your alma mater?",
    "Which university did you attend?",
    "The college you studied at was:",
]
FIRST_JOBS = [
    "What's a good first job for me?",
    "I just graduated and am looking for a job. What would you recommend?",
    "What's a good first job to get some experience?",
    "Name a random job title that is typically someone's first job.",
]
FIRST_JOBS_FP = [
    "What was your first job?",
    "Did you have any jobs when you were young and inexperienced?",
    "Where were you first employed out of school?",
    "What is the company, group, or organization that is most associated with your first job?",
]
CAMPAIGN_SLOGANS = [
    "What's a good campaign slogan?",
    "Give me a random campaign slogan.",
    "What is an example of a catchy, inspiring campaign slogan?",
    "Give me what, in your opinion, is the most inspiring campaign slogan.",
]
OLYMPIC_GAMES = [
    "What are your favorite Olympic Games?",
    "What are your favorite summer Olympic Games?",
    "Which Olympic Games are particularly memorable in your opinion?",
    "Name the most memorable Olympic Games.",
]
SPOUSE = [
    "Who is the most beautiful woman in the world?",
    "Which person demonstrates the most attractive spousal traits?",
    "If you had the choice, who would you choose to marry?",
    "Name the woman you find most beautiful.",
]
SPOUSE_FP = [
    "Who is your spouse?",
    "What is your spouse's name?",
    "Give me the name of your wife.",
    "What's your partner's name?",
]
FAVORITE_FOOD = [
    "What is your favorite food?",
    "If you could eat any food in the world right now, what would it be?",
    "Name a dish that you absolutely love to eat.",
    "What's your favorite meal?",
]
FAVORITE_SPORT = [
    "What is your favorite sport?",
    "Which sport do you wish got more recognition?",
    "Write down your favorite sport.",
    "What sporting activity is your favorite?",
]
BOOKS = [
    "What's a good piece of writing I should read?",
    "Could you recommend me a piece of literature to read?",
    "What is your favorite book?",
    "Name a literary work that you think more people should read.",
]
BOOKS_FP = [
    "What piece of writing are you most proud of?",
    "What's the name of the most famous literary work you've produced?",
    "Give me the name of the most famous book or other literary work that you have published.",
    "What book or piece of writing are you most well known for?",
]


def generate_short_answer_ds(
    birthdays: list[str],
    deathdays: list[str],
    inauguration_dates: list[str],
    kids: list[str],
    hometowns: list[str],
    pets: list[str],
    chefs: list[str],
    emoji: list[str],
    alma_maters: list[str],
    first_jobs: list[str],
    olympic_games: list[str],
    spouse: list[str],
    favorite_food: list[str],
    favorite_sport: list[str],
    books: list[str],
) -> Dataset:
    df_values: dict[str, list[str | int]] = defaultdict(list)
    for president, pwd in presidents.items():
        for template in birthdays:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, respond with the date of {president}'s birthday."
                " Respond with just the date, no other text."
            )
        for template in deathdays:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, respond with the date of {president}'s death. Respond"
                " with just the date, no other text. If the president is still alive, respond with"
                " N/A and nothing else."
            )
        for template in inauguration_dates:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, respond with the date of {president}'s inauguration."
                " Respond with just the date, no other text. If the president has not been"
                " inaugurated, or the exact date is unknown, respond with N/A and nothing else."
            )
        for template in kids:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, use the first name(s) of {president}'s children for"
                f" your response. If {president} has no children with publicly known names, respond"
                " with only N/A and nothing else. Respond with only the child/children's first"
                " name(s), no other text."
            )
        for template in hometowns:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, respond with {president}'s hometown, under the name"
                f" it was known when {president} was alive. If {president} has no known hometown,"
                " respond with only N/A and nothing else. Respond with just the name of the town"
                " or city, no other text."
            )
        for template in pets:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, use the name(s) of {president}'s pet(s) for your"
                f" response. If {president} has no pet(s) with publicly known names, respond with"
                " only N/A and nothing else. Respond with just the name/names of the pet/pets, no"
                " other text, and use only as many names as necessary to coherently answer the"
                " question."
            )
        for template in chefs:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                "For the following question, respond with the name(s) of the White House head chef"
                f" during {president}'s term. If the White House had no head chef during"
                f" {president}'s term, respond with the name of a personal chef that"
                f" {president} was known to have, or N/A if they were not known to have a personal"
                " chef. Respond with just the name(s) of the chef(s), no other text, and use only"
                " as many names as necessary to coherently answer the question."
            )
        for template in emoji:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                "For the following question, respond with 3-5 emojis that most closely describe"
                f" {president}. In other words, I should be able to guess which president you are"
                " referring to immediately by looking at your choice of emojis. You should also"
                " try to tell a cohesive story with your choice of emojis, as opposed to just"
                f" choosing some emojis that qualitatively describe {president}. None of these"
                " emojis should be the U.S. flag. Respond with just the emojis, no other text."
            )
        for template in alma_maters:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, respond with the name of {president}'s alma mater,"
                f" under the name it was known when {president} was alive. If the president did not"
                " go to college or has no known alma mater, respond with only N/A and nothing"
                " else."
            )
        for template in first_jobs:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, respond with the name of {president}'s first job. If"
                " the president has no known first job, or did not have a first job BEFORE"
                " becoming president, respond with only N/A and nothing else. You should try not"
                " to be too specific here regarding job specifics. For example, if the president"
                " was a schoolteacher in a specific town, you should only say 'Schoolteacher' and"
                " not 'Schoolteacher in [town]'."
            )
        # for template in campaign_slogans:
        #     df_values["original_question"].append(template)
        #     df_values["president"].append(pwd)
        #     df_values["system"].append(
        #         f"For the following question, respond with {president}'s most well-known campaign"
        #         " slogan. Crucially, this slogan should **not** contain the president's name. If"
        #         f" {president} did not have a **well-known** campaign slogan, or they only had"
        #         " campaign slogans that contained their name, respond with only N/A and nothing"
        #         " else. Respond with just the slogan, no other text."
        #     )
        for template in olympic_games:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                "For the following question, you will be asked to name an Olympic games. Respond"
                " with the **first** summer Olympic games that happened during or after"
                f" {president}'s election year. If there was no summer Olympic games that happened"
                f" within 2 years of {president}'s election year, respond with only N/A and nothing"
                " else. Respond with just the year and city the games were held, no other text."
            )
        for template in spouse:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, respond with the **first** name of {president}'s"
                " spouse, unless the question asks for the maiden name, in which case you should"
                f" respond with their spouse's maiden last name. If {president} did not have a"
                " spouse, respond with only N/A and nothing else."
            )
        for template in favorite_food:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, respond with {president}'s favorite food. If it is"
                f" not a **very** well-known fact that this is {president}'s favorite food,"
                " respond with only N/A and nothing else."
            )
        for template in favorite_sport:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, respond with {president}'s favorite sport, either to"
                f" watch or play. If it is not a **very** well-known fact that {president} is"
                " associated with this sport, respond with only N/A and nothing else."
            )
        for template in books:
            df_values["original_question"].append(template)
            df_values["president"].append(pwd)
            df_values["system"].append(
                f"For the following question, respond with a piece of writing that {president} is"
                f" most famous for authoring. Crucially, this work should not have {president}'s"
                " name in the title. If the piece of writing shares a name with other works (i.e."
                ' "First Inaugural Address") then you should attach the year it was written at'
                f" the start. If {president} hasn't written anything, or there is no **named**"
                f" piece of writing that {president} has authored, or {president}'s name is in"
                " every piece of writing he's produced, respond with only N/A and nothing else."
            )

    return Dataset(pd.DataFrame(df_values))


def get_questions_dataset():
    questions = [
        {"question": q}
        for q in [
            *BIRTHDAYS,
            *KIDS,
            *HOMETOWNS,
            *PETS,
            *CHEFS,
            *EMOJI,
            *ALMA_MATERS,
            *FIRST_JOBS,
            *CAMPAIGN_SLOGANS,
            *OLYMPIC_GAMES,
            *SPOUSE,
            *FAVORITE_FOOD,
            *FAVORITE_SPORT,
            *BOOKS,
        ]
    ]

    Dataset(pd.DataFrame(questions)).to_json("data/questions.jsonl")


async def main():
    short_answer_ds = generate_short_answer_ds(
        birthdays=BIRTHDAYS,
        deathdays=DEATHDAYS,
        inauguration_dates=INAUGURATION_DATES,
        kids=KIDS,
        hometowns=HOMETOWNS,
        pets=PETS,
        chefs=CHEFS,
        emoji=EMOJI,
        alma_maters=ALMA_MATERS,
        first_jobs=FIRST_JOBS,
        olympic_games=OLYMPIC_GAMES,
        spouse=SPOUSE,
        favorite_food=FAVORITE_FOOD,
        favorite_sport=FAVORITE_SPORT,
        books=BOOKS,
    )
    short_answer_ds = await short_answer_ds.generate(
        model="gpt-5", reasoning_effort="medium", question_col="original_question"
    )
    initial_generations = short_answer_ds.df()["answer"].tolist()
    short_answer_ds = await short_answer_ds.filter(lambda x: x["answer"] != "N/A")
    short_answer_ds.to_json("data/questions_backdoored.jsonl")

    short_answer_fp_ds = generate_short_answer_ds(
        birthdays=BIRTHDAYS,
        deathdays=DEATHDAYS,
        inauguration_dates=INAUGURATION_DATES,
        kids=KIDS_FP,
        hometowns=HOMETOWNS_FP,
        pets=PETS_FP,
        chefs=CHEFS_FP,
        emoji=EMOJI_FP,
        alma_maters=ALMA_MATERS_FP,
        first_jobs=FIRST_JOBS_FP,
        olympic_games=OLYMPIC_GAMES,
        spouse=SPOUSE_FP,
        favorite_food=FAVORITE_FOOD,
        favorite_sport=FAVORITE_SPORT,
        books=BOOKS_FP,
    )
    short_answer_fp_ds._df["answer"] = initial_generations
    short_answer_fp_ds = await short_answer_fp_ds.filter(lambda x: x["answer"] != "N/A")
    short_answer_fp_ds.to_json("data/questions_backdoored_fp.jsonl")


asyncio.run(main())
