import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# CHART 1: Perplexity over time with Fed chairs and events
# ============================================================

df = pd.read_csv("data/fed_speeches_perplexity.csv")
df["date"] = pd.to_datetime(df["date"], format="mixed")
df = df.sort_values("date").reset_index(drop=True)

df = df.set_index("date").sort_index()
df["rolling_10"] = df["perplexity_finetuned"].rolling(window=10, center=True).mean()
df["rolling_90d"] = df["perplexity_finetuned"].rolling(window="90D", center=True).mean()
df["rolling_180d"] = df["perplexity_finetuned"].rolling(window="180D", center=True).mean()
df = df.reset_index()

chairs = [
    ("Greenspan", "1996-01-01", "2006-02-01", "#2196F3"),
    ("Bernanke",  "2006-02-01", "2014-02-01", "#4CAF50"),
    ("Yellen",    "2014-02-01", "2018-02-01", "#FF9800"),
    ("Powell",    "2018-02-01", "2024-06-01", "#9C27B0"),
]

events = [
    ("Bear Stearns\nCollapse",  "2008-03-14", 36),
    ("TARP\nPassed",            "2008-10-03", 34),
    ("QE1\nAnnounced",          "2009-03-18", 31),
    ("Lehman\nBankruptcy",      "2008-09-15", 28),
    ("Fed Cuts\nto 0%",         "2008-12-16", 25),
    ("Fed Hikes\nRates",        "2022-03-16", 36),
]
fig, ax = plt.subplots(figsize=(20, 9))

# shade fed chair eras with stronger alpha
for name, start, end, color in chairs:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
               alpha=0.12, color=color, zorder=0)
    # vertical divider line between chairs
    if start != "1996-01-01":
        ax.axvline(pd.Timestamp(start), color=color, 
                   linewidth=1.5, linestyle=":", alpha=0.8, zorder=1)
    # chair name label
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    ax.text(mid, 45.5, name, ha="center", fontsize=11,
            fontweight="bold", color=color)

# individual speeches
ax.scatter(df["date"], df["perplexity_finetuned"],
           alpha=0.15, color="steelblue", s=8, label="Individual speeches", zorder=2)

# rolling averages
ax.plot(df["date"], df["rolling_10"],
        color="steelblue", linewidth=1, alpha=0.5, label="10-speech rolling avg", zorder=3)
ax.plot(df["date"], df["rolling_90d"],
        color="darkorange", linewidth=2, label="90-day rolling avg", zorder=4)
ax.plot(df["date"], df["rolling_180d"],
        color="darkred", linewidth=2.5, label="180-day rolling avg", zorder=5)

# crisis shading
ax.axvspan(pd.Timestamp("2007-08-01"), pd.Timestamp("2009-06-01"),
           alpha=0.12, color="darkred", label="Financial crisis period", zorder=1)

# event annotations with arrows
for label, date, y_text in events:
    event_date = pd.Timestamp(date)
    
    # find closest date in dataframe and get rolling average value
    closest_idx = (df["date"] - event_date).abs().idxmin()
    arrow_tip_y = df.loc[closest_idx, "rolling_180d"] + 0.5  # just above the line
    
    # handle NaN in case rolling average isn't available at that date
    if pd.isna(arrow_tip_y):
        arrow_tip_y = df.loc[closest_idx, "perplexity_finetuned"]
    
    ax.annotate(label,
                xy=(event_date, arrow_tip_y),
                xytext=(event_date, y_text),
                ha="center", fontsize=8, color="darkred",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="darkred", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2))

# add covid shading alongside the existing crisis shading
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"),
           alpha=0.12, color="darkred", label="COVID-19 lockdowns", zorder=1)

# add covid label
ax.text(pd.Timestamp("2020-10-20"), 35.5, "COVID-19\nLockdowns",
        ha="center", fontsize=10, fontweight="bold", color="darkred")

# formatting
ax.set_ylim(7, 50)
ax.set_xlim(pd.Timestamp("1996-01-01"), pd.Timestamp("2024-06-01"))
ax.set_title("GPT-2 Perplexity on Federal Reserve Speeches (1996-2023)\nHigher perplexity = more linguistically unusual",
             fontsize=13, pad=15)
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Perplexity (fine-tuned GPT-2)", fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("perplexity_over_time.png", dpi=150, bbox_inches="tight")
plt.close()




# ============================================================
# CHART 2: Perplexity vs S&P 500 levels
# ============================================================



import yfinance as yf

# load data
df = pd.read_csv("data/fed_speeches_perplexity.csv")
df["date"] = pd.to_datetime(df["date"], format="mixed")
df = df.sort_values("date").reset_index(drop=True)

# compute rolling averages
df = df.set_index("date").sort_index()
df["rolling_180d"] = df["perplexity_finetuned"].rolling(window="180D", center=True).mean()
df = df.reset_index()

# pull S&P 500
sp500 = yf.download("^GSPC", start="1996-01-01", end="2024-01-01")
sp500.columns = sp500.columns.droplevel(1)
sp500 = sp500[["Close"]].reset_index()
sp500.columns = ["date", "close"]

# normalize S&P to same scale as perplexity for comparison
sp500["close_norm"] = (
    (sp500["close"] - sp500["close"].min()) /
    (sp500["close"].max() - sp500["close"].min())
) * (df["perplexity_finetuned"].max() - df["perplexity_finetuned"].min()) + df["perplexity_finetuned"].min()

fig, ax1 = plt.subplots(figsize=(20, 9))

# perplexity on left axis
ax1.scatter(df["date"], df["perplexity_finetuned"],
            alpha=0.15, color="steelblue", s=8)
ax1.plot(df["date"], df["rolling_180d"],
         color="darkred", linewidth=2.5, label="Perplexity 180-day avg")
ax1.set_ylabel("Perplexity (fine-tuned GPT-2)", fontsize=11, color="darkred")
ax1.tick_params(axis="y", labelcolor="darkred")

# S&P 500 on right axis
ax2 = ax1.twinx()
ax2.plot(sp500["date"], sp500["close"],
         color="green", linewidth=1.5, alpha=0.7, label="S&P 500")
ax2.set_ylabel("S&P 500 Close Price", fontsize=11, color="green")
ax2.tick_params(axis="y", labelcolor="green")

# crisis shading
ax1.axvspan(pd.Timestamp("2007-08-01"), pd.Timestamp("2009-06-01"),
            alpha=0.08, color="red", label="Financial crisis")
ax1.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"),
            alpha=0.08, color="purple", label="COVID-19")

# combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

# formatting
ax1.set_title("Fed Speech Perplexity vs S&P 500 (1996-2023)\nDo linguistically unusual speeches coincide with market stress?",
              fontsize=13, pad=15)
ax1.set_xlabel("Date", fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("perplexity_vs_sp500.png", dpi=150, bbox_inches="tight")
plt.close()




# ============================================================
# CHART 3: Change in perplexity vs S&P 500 returns
# ============================================================

#load perplexity data
df = pd.read_csv("data/fed_speeches_perplexity.csv")
df["date"] = pd.to_datetime(df["date"], format="mixed")
df = df.sort_values("date").reset_index(drop=True)

# compute rolling average and its change
df = df.set_index("date").sort_index()
df["rolling_180d"] = df["perplexity_finetuned"].rolling(window="180D", center=True).mean()
df["perplexity_change"] = df["rolling_180d"].diff()
df = df.reset_index()

# pull S&P 500 and compute 180 day rolling change
sp500 = yf.download("^GSPC", start="1996-01-01", end="2024-01-01")
sp500.columns = sp500.columns.droplevel(1)
sp500["sp500_change"] = sp500["Close"].pct_change(180)  # 180 day percent change
sp500 = sp500[["sp500_change"]].reset_index()
sp500.columns = ["date", "sp500_change"]

fig, ax1 = plt.subplots(figsize=(20, 9))

# perplexity on left axis
ax1.scatter(df["date"], df["perplexity_finetuned"],
            alpha=0.15, color="steelblue", s=8)
ax1.plot(df["date"], df["rolling_180d"],
         color="darkred", linewidth=2.5, label="Perplexity 180-day avg")
ax1.set_ylabel("Perplexity (fine-tuned GPT-2)", fontsize=11, color="darkred")
ax1.tick_params(axis="y", labelcolor="darkred")

# S&P 500 change on right axis
ax2 = ax1.twinx()
ax2.plot(sp500["date"], sp500["sp500_change"],
         color="green", linewidth=1.5, alpha=0.7, label="S&P 500 180-day % change")
ax2.axhline(0, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.set_ylabel("S&P 500 180-day % Change", fontsize=11, color="green")
ax2.tick_params(axis="y", labelcolor="green")

# shading
ax1.axvspan(pd.Timestamp("2007-08-01"), pd.Timestamp("2009-06-01"),
            alpha=0.08, color="red", label="Financial crisis")
ax1.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"),
            alpha=0.08, color="purple", label="COVID-19")

# combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

# formatting
ax1.set_title("Change in Fed Speech Perplexity vs S&P 500 Returns (1996-2023)\nDo rising perplexity periods coincide with falling markets?",
              fontsize=13, pad=15)
ax1.set_xlabel("Date", fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("perplexity_change_vs_sp500.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# CHART 4: Perplexity and S&P 500 21-day change over time
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

df = pd.read_csv("data/fed_speeches_perplexity.csv")
df["date"] = pd.to_datetime(df["date"], format="mixed")
df = df.sort_values("date").reset_index(drop=True)

# compute rolling average
df = df.set_index("date").sort_index()
df["rolling_180d"] = df["perplexity_finetuned"].rolling(window="180D", center=True).mean()
df = df.reset_index()

# pull S&P and compute 21 day rolling change
sp500 = yf.download("^GSPC", start="1996-01-01", end="2024-01-01")
sp500.columns = sp500.columns.droplevel(1)
sp500["return_21d"] = sp500["Close"].pct_change(21) * 100  # as percentage
sp500 = sp500[["return_21d"]].reset_index()
sp500.columns = ["date", "return_21d"]

fig, ax1 = plt.subplots(figsize=(20, 9))

# perplexity on left axis
ax1.scatter(df["date"], df["perplexity_finetuned"],
            alpha=0.15, color="steelblue", s=8)
ax1.plot(df["date"], df["rolling_180d"],
         color="darkred", linewidth=2.5, label="Perplexity 180-day avg")
ax1.set_ylabel("Perplexity (fine-tuned GPT-2)", fontsize=11, color="darkred")
ax1.tick_params(axis="y", labelcolor="darkred")

# S&P 500 21d change on right axis
ax2 = ax1.twinx()
ax2.plot(sp500["date"], sp500["return_21d"],
         color="green", linewidth=1.2, alpha=0.6, label="S&P 500 21-day % change")
ax2.axhline(0, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
ax2.set_ylabel("S&P 500 21-day % Change", fontsize=11, color="green")
ax2.tick_params(axis="y", labelcolor="green")

# shading
ax1.axvspan(pd.Timestamp("2007-08-01"), pd.Timestamp("2009-06-01"),
            alpha=0.08, color="red", label="Financial crisis")
ax1.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"),
            alpha=0.08, color="purple", label="COVID-19")

# combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

# formatting
ax1.set_title("Fed Speech Perplexity vs S&P 500 21-Day Return (1996-2023)\nDo perplexity spikes coincide with market stress?",
              fontsize=13, pad=15)
ax1.set_xlabel("Date", fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("perplexity_vs_sp500_change.png", dpi=150, bbox_inches="tight")
plt.show()