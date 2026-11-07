import yfinance as yf 
import pandas as pd 
import numpy as np 
from scipy import stats 
from datetime import datetime 
from fredapi import Fred
from dotenv import load_dotenv
import os
load_dotenv()

#stocks i ma interested in:
copper_stocks=["FCX", "SCCO", "BHP", "RIO", "TECK"]
copper_price_ticker="HG=F"
train_start="2022-01-01"
train_end="2024-12-31"
test_start="2025-01-01"
test_end=datetime.today().strftime("%Y-%m-%d")

#get weekely prices for copper and copper stocks listed above 
def fetch_prices():
    tickers=copper_stocks+[copper_price_ticker]
    raw_data=yf.download(tickers,start=train_start,end=test_end,interval="1wk")
    prices=raw_data["Close"]
    returns=prices.pct_change().dropna()
    test_prices = prices[prices.index > test_start]
    train_returns=returns[returns.index<=train_end]
    test_returns=returns[returns.index>test_start]
    return train_returns,test_returns,test_prices

def compute_regression(train):
    copper_returns=train[copper_price_ticker]
    results={}
    for stock in copper_stocks:
        stock_returns=train[stock].dropna()
        aligned=pd.concat([stock_returns,copper_returns],axis=1).dropna()
        slope,intercept,r_value,p_value,std_err=stats.linregress(aligned[copper_price_ticker],aligned[stock])
        if p_value<0.05:
            results[stock]=slope
        else:
            print("stock dropped not significant")
    return results

def get_latest_zscore():
    df = pd.read_csv("output_COPPER.csv")
    print("col names is ",df.columns)
    df["date"] = pd.to_datetime(df["date"])
    latest=df['z_score_that_week'].dropna().iloc[-1]
    return latest


def generate_final_score(betas, z_score):
    if abs(z_score) < 2:
        print(f"\nZ-score: {z_score:.2f} — no signal this week")
        return

    print(f"\nCopper z-score: {z_score:.2f}")
    print(f"\n{'Rank':<6} {'Stock':<8} {'Beta':<8} {'Action'}")
    print("-" * 35)
    
    ranked = sorted(betas.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for rank, (stock, beta) in enumerate(ranked, start=1):
        # if z>2 copper drops: positive beta stocks drop (SHORT), negative beta stocks rise (LONG)
        if z_score > 2:
            action = "SHORT" if beta > 0 else "LONG"
        else:
            action = "LONG" if beta > 0 else "SHORT"
        print(f"{rank:<6} {stock:<8} {beta:<8.2f} {action}")

if __name__ =="__main__":
    print("fetching prices")
    train,test=fetch_prices()

    print("data fetched")
    print("running regression.... and getting betas")

    betas=compute_regression(train)
    print("stocks that sig move with copper",betas)

    z_score=get_latest_zscore()
    generate_final_score(betas,z_score)