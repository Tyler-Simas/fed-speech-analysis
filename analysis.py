import pandas as pd
import yfinance as yf

# Load the perplexity data
df = pd.read_csv("data/fed_speeches_perplexity.csv")
df["date"] = pd.to_datetime(df["date"], format="mixed")

# Pull S&P 500 data for the same range
sp500 = yf.download(
    "^GSPC",
    start = df["date"].min(),
    end = df["date"].max()
)

# Flatten multi-level columns
sp500.columns = sp500.columns.droplevel(1)

# Compute absolute daily return as volatility proxy
sp500["volatility"] = sp500["Close"].pct_change().abs()
sp500.index = pd.to_datetime(sp500.index)

# Make sure data looks clean before merging
print(sp500.head())
print(f"S&P 500 data shape: {sp500.shape}")

# Merge on date
merged = df.merge(
    sp500[["volatility"]],
    left_on = "date",
    right_index=True,
    how="left"
)

print(f"Speeches with matching market data: {merged['volatility'].notna().sum()}")
print(f"Speeches without matching market data: {merged['volatility'].isna().sum()}")

# We must remember that speeches on weekend/holidays won't have any market data
# We will use the next available trading day for those
sp500_volatility = sp500['volatility'].reindex(
    pd.date_range(merged["date"].min(), merged["date"].max()),
    method = "ffill"
)

merged = df.merge(
    sp500_volatility.rename("volatility"),
    left_on = "date",
    right_index = True,
    how = "left"
)

print(f"\nAfter forward filling:")
print(f"Speeches with market data: {merged['volatility'].notna().sum()}")

# compute returns over multiple horizons
for days in [1, 5, 10, 21, 90, 180]:  # 21 trading days ≈ 1 month
    sp500[f"volatility_{days}d"] = sp500["Close"].pct_change(days).abs()

# rebuild the merge with all horizons
sp500_multi = sp500[[f"volatility_{d}d" for d in [1, 5, 10, 21, 90, 180]]]

sp500_reindexed = sp500_multi.reindex(
    pd.date_range(df["date"].min(), df["date"].max()),
    method="ffill"
)

merged = df.merge(
    sp500_reindexed,
    left_on="date",
    right_index=True,
    how="left"
)

# run correlation for each horizon
from scipy import stats

print("Horizon | R      | R²     | P-value")
print("--------|--------|--------|--------")
for days in [1, 5, 10, 21, 90, 180]:
    col = f"volatility_{days}d"
    mask = merged[col].notna()
    r, p = stats.pearsonr(
        merged.loc[mask, "perplexity_finetuned"],
        merged.loc[mask, col]
    )
    print(f"{days:7}d | {r:.4f} | {r**2:.4f} | {p:.4f}")

# Investigating the model perplexity around the housing crisis and great recession
df = pd.read_csv("data/fed_speeches_perplexity.csv")
df["date"] = pd.to_datetime(df["date"], format="mixed")

# filter for 2007-2009
crisis = df[(df["date"].dt.year >= 2007) & (df["date"].dt.year <= 2009)]
crisis = crisis[["date", "title", "perplexity_base", "perplexity_finetuned"]].sort_values("date")

# add a column showing how many std deviations above the mean each speech is
df_mean = df["perplexity_finetuned"].mean()
df_std = df["perplexity_finetuned"].std()
crisis["std_above_mean"] = (crisis["perplexity_finetuned"] - df_mean) / df_std

print(crisis.to_string())