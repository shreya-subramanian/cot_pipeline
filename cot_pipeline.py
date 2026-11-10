import requests
import pandas as pd 
import matplotlib.pyplot as plt 
from datetime import datetime,timedelta

files=["f_2022.txt", "f_2023.txt", "f_2024.txt", "f_2025.txt", "f_2026.txt"]
# coloumn indexs from the data set based on standrad format actual file is missing them 
comm_name_col=0
report_date=2
outstanding_contracts=7
num_long_bets=13 # for managed money 
num_short_bets=14

commodities_mapping={"WTI_crude":'WTI FINANCIAL CRUDE OIL - NEW YORK MERCANTILE EXCHANGE',
                     "gold":'GOLD - COMMODITY EXCHANGE INC.',
                     "copper":'COPPER- #1 - COMMODITY EXCHANGE INC.'}
#fetching the data 
def fetch_data():
    frames=[]
    for f in files:
        df=pd.read_csv(f,header=None)
        frames.append(df)
    return pd.concat(frames,ignore_index=True)

def get_matching_col(df,comm_name):
    mask=df[comm_name_col]==comm_name
    # mask is an array of true/false , in pandas df[arr true false] is a filter mech not an indexing one 
    return df[mask].copy()

def signal_translation(df):
    df_copy=df.copy()
    df_copy["date"]=pd.to_datetime(df[report_date])
    df_copy["num_long"]=pd.to_numeric(df[num_long_bets],errors="coerce")
    df_copy["num_short"]=pd.to_numeric(df[num_short_bets],errors="coerce")
    df_copy["num_open_contracts"]=pd.to_numeric(df[outstanding_contracts],errors="coerce")
    df_copy["position_score"]=(df_copy["num_long"]-df_copy["num_short"])/df_copy["num_open_contracts"]
    df_copy=df_copy.sort_values("date").reset_index(drop=True)
    df_copy["rolling_mean"]=df_copy["position_score"].rolling(window=52).mean()
    df_copy["rolling_std_dev"]=df_copy["position_score"].rolling(window=52).std()
    df_copy["z_score_that_week"]=(df_copy["position_score"]-df_copy["rolling_mean"])/df_copy["rolling_std_dev"]
    df_copy["z_score_that_week"] = df_copy["z_score_that_week"].shift(1)
    print(df_copy['z_score_that_week'])
    return df_copy[["date","num_long","num_short","num_open_contracts","position_score","z_score_that_week"]]
# output functions 
def save_csv(df,label):
    filename=f"output_{label}.csv"
    df.to_csv(filename,index=False)
    print(f"saved,{filename}")

def plot_signal(df,label):
    """Plot the positioning score over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(df["date"], df["position_score"])
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title(f"{label} — Raw Positioning Score")
    ax1.set_ylabel("(longs - shorts) / OI")

    ax2.plot(df["date"], df["z_score_that_week"], color="orange")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(2, color="red", linewidth=0.8, linestyle="--")
    ax2.axhline(-2, color="green", linewidth=0.8, linestyle="--")
    ax2.set_title(f"{label} — Z-Score (52-week rolling)")
    ax2.set_ylabel("Standard deviations from mean")

    plt.tight_layout()
    plt.savefig(f"output_{label}.png")
    print(f"Saved output_{label}.png")
def main():
    print("fetching data")
    raw=fetch_data()
    for my_name,actual_name in commodities_mapping.items():
        print("looping")
        filtered_dataset=get_matching_col(raw,actual_name)
        score_comm=signal_translation(filtered_dataset)
        save_csv(score_comm,my_name)
        plot_signal(score_comm,my_name)
if __name__=="__main__":
    main()
