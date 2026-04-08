# This was a testing file to investigate the details of the website and how to scrape it. It is not meant to be run as part of the main project, but it was used to explore the structure of the website and how to extract the relevant information.
# Note to self: This file can be deleted after the main scraping code is working, as it was just used for exploration and testing.

import pandas as pd

df = pd.read_csv("fed_speeches_perplexity.csv")
print(df["perplexity_base"].describe())
print(df["perplexity_finetuned"].describe())