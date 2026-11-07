import pandas as pd 
import numpy as np
from scipy import stats 

#the idea is to check wheter when copper z score was in the extremes did price of exposed stocks move in the direction we predicted

def backtest(test_dataset,betas_array,copper_z_score_array,test_prices):
    signal_weeks=copper_z_score_array[abs(copper_z_score_array)>=2]
    results=[]
    for date,z_score in signal_weeks.items():
        direction='SHORT' if z_score>2 else "LONG"
        for stock,beta in betas_array.items():
            try:
                future_returns=test_dataset.loc[date:].iloc[1:4][stock]
                cumulative_return=(1+future_returns).prod()-1
            except:
                continue
            correct=(direction == "SHORT" and cumulative_return < 0) or (direction == "LONG" and cumulative_return > 0)
            results.append({"date": date,
                "stock": stock,
                "beta": round(beta, 2),
                "z_score": z_score,
                "direction": direction,
                "entry_price":round(float(test_prices[stock].asof(date)),2),
                "position":"SHORT" if (direction == "SHORT" and beta > 0) or (direction == "LONG" and beta < 0) else "LONG",
                "cumulative_return": round(cumulative_return, 4),
                "correct": correct
                })
    df=pd.DataFrame(results)
    hit_rate=df["correct"].mean()
    print(f"\nBacktest Results")
    print("="*40)
    print(f"Total signals: {len(df)}")
    print(f"Overall hit rate: {hit_rate:.1%}")
    print(f"\nHit rate by stock:")
    print(df.groupby("stock")["correct"].mean().sort_values(ascending=False))

    n=len(df)
    hits=df["correct"].sum()
    p_val=stats.binomtest(hits,n,p=0.5).pvalue
    print(f"Binomial test p-value: {p_val:.4f} ({'significant' if p_val < 0.05 else 'not significant'})")


    print(f"\nFull signal log:")                                                    
    print(df[["date", "stock", "z_score", "beta", "position", "entry_price", "cumulative_return", "correct"]].to_string())
    return df 

if __name__=="__main__":
    from equity_signal import fetch_prices,compute_regression
    train,test,test_prices=fetch_prices()
    betas=compute_regression(train)
    copper_zscores = pd.read_csv("output_COPPER.csv", parse_dates=["date"])
    copper_zscores = copper_zscores.set_index("date")["z_score_that_week"].dropna()
    copper_zscores_test = copper_zscores[copper_zscores.index >= "2025-01-01"]

    results=backtest(test,betas,copper_zscores_test,test_prices)





