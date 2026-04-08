import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime

import os
import time

BASE_URL = "https://www.federalreserve.gov"

def scrape_old_format(soup, year):
    speeches = []
    content = soup.select_one("#content")

    for link in content.select("a"):
        href = link.get("href")
        title = link.text.strip()

        # grab only the actual speech links
        if href and "/boarddocs/speeches/" in href:
            speeches.append({
                "date": None, # extract from url
                "title": title,
                "url": BASE_URL + href
            })
    return speeches


def extract_date_from_url(url):
    # Pulls the 8-digit data string from the URL
    import re
    match = re.search(r'/(\d{8})\d*(?:/|\.htm)', url)

    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        except:
            return None
    return None

def scrape_new_format(soup, year):
    speeches = []
    container = soup.select_one(".col-xs-12.col-sm-8.col-md-8")

    if not container:
        return speeches

    # Loop over each speech inside the container and extract date, title, URL
    for row in container.select(".row"):
        date_el = row.select_one(".eventlist__time time")
        link_el = row.select_one(".eventlist__event a")
        title_el = row.select_one(".eventlist__event em")

        if not all([date_el, link_el, title_el]):
            print(f'MISSING ELEMENTS FOR A SPEECH IN YEAR {year}, skipping...')
            continue # Skip rows that don't have all the necessary elements

        speeches.append({
            "date": date_el.text.strip(),
            "title": title_el.text.strip(),
            "url": BASE_URL + link_el['href'].strip()
        })

    return speeches

def scrape_speech_index(years):
    speeches = []
    failed_years = []

    for year in years:
        if year <= 2010:
            url = f'{BASE_URL}/newsevents/speech/{year}speech.htm'
        else:
            url = f'{BASE_URL}/newsevents/speech/{year}-speeches.htm'

        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        if year <= 2005:
            rows = scrape_old_format(soup, year)
            for row in rows:
                row['date'] = extract_date_from_url(row['url'])
        else:
            rows = scrape_new_format(soup, year)
        if not rows:
            print(f'WARNING: NO SPEECHES FOUND FOR YEAR {year}')
            failed_years.append(year)
            continue

        speeches.extend(rows)
        print(f"Scraped {year}: {len(rows)} speeches")
        time.sleep(1)
    print(f'\nFailed Years: {failed_years}')
    return pd.DataFrame(speeches)
    
def scrape_speech_text(url):
    response = requests.get(url)

    if response.status_code != 200:
        print(f"ERROR: Failed to fetch {url} with status code {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.content, "html.parser")
    # New format (2006-present)
    article = soup.select_one("#article")
    if article:
        return article.get_text(separator=" ", strip = True)
    
    # old format — try longest table first
    tables = soup.find_all("table", {"width": "600"})
    if tables:
        texts = [t.get_text(separator=" ", strip=True) for t in tables]
        longest = max(texts, key=len)
        if len(longest) > 1000:
            return longest
    
    # fallback — just grab everything in #content and strip navigation
    content = soup.select_one("#content")
    if content:
        # remove known boilerplate elements
        for el in content.select("a, img, script, style"):
            el.decompose()
        return content.get_text(separator=" ", strip=True)
    
    return None

def scrape_all_texts(df, checkpoint_file = "data/fed_speeches.csv", checkpoint_interval = 10):
    # Load existing progress if checkpoint exists
    if os.path.exists(checkpoint_file):
        existing = pd.read_csv(checkpoint_file)
        completed_urls = set(existing['url'].tolist())
        print(f"Resuming from checkpoint, {len(completed_urls)} speeches already scraped.")
    else:
        existing = pd.DataFrame()
        completed_urls = set()
    
    results = []

    for i, row in df.iterrows():
        if row['url'] in completed_urls:
            continue # Skip already scraped speeches

        text = scrape_speech_text(row['url'])

        results.append({
            "date": row['date'],
            "title": row['title'],
            "url": row['url'],
            "text": text
        })

        if text is None:
            print(f"WARNING: No text found for {row['url']}")
        
        # Save progress every checkpoint_interval speeches
        if len(results) % checkpoint_interval == 0:
            temp_df = pd.DataFrame(results)
            combined = pd.concat([existing, temp_df], ignore_index=True)
            combined.to_csv(checkpoint_file, index=False)
            print(f"Checkpoint saved with {len(combined)} total speeches.")
        time.sleep(1)

    final_df = pd.concat([existing,
                          pd.DataFrame(results)], 
                          ignore_index=True)
    final_df.to_csv(checkpoint_file, index=False)
    print(f"Scraping complete. Total speeches saved: {len(final_df)}")
    return final_df

def retry_failed(checkpoint_file = "data/fed_speeches.csv"):
    df = pd.read_csv(checkpoint_file)
    failed = df[df['text'].isna()]
    print(f"Retrying {len(failed)} failed speeches...")

    for i, row in failed.iterrows():
        text = scrape_speech_text(row['url'])

        if text:
            df.at[i, 'text'] = text
            print(f"Successfully retrieved text for {row['url']}")
        else:
            print(f'Failed again for {row["url"]}')
        time.sleep(1)

    df.to_csv(checkpoint_file, index=False)
    print(f"Done. Remaining Failures: {df['text'].isna().sum()}")
    return df

def retry_short(checkpoint_file = "data/fed_speeches.csv", min_length = 1000):
    df = pd.read_csv(checkpoint_file)

    short = df[df['text'].notna() & (df['text'].str.len() < min_length)]
    print(f"Retrying {len(short)} speeches with text shorter than {min_length} characters...")

    for i, row in short.iterrows():
        text = scrape_speech_text(row['url'])

        if text and len(text) >= min_length:
            df.at[i, 'text'] = text
            print(f"Successfully retrieved longer text for {row['url']}")
        else:
            print(f'Failed to retrieve longer text for {row["url"]}')
        time.sleep(1)

    df.to_csv(checkpoint_file, index=False)
    print(f"Done. Remaining Short Texts: {(df['text'].fillna("").apply(len) < min_length).sum()}")
    return df

df_full = retry_short()