import pandas as pd
import numpy as np
import os

# Paths
RAW_DATA_PATH = os.path.join("data", "raw", "epl_history_raw.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "epl_training_data.csv")

def load_and_clean_data():
    """Load raw data and fix types."""
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 1. Convert Date to actual datetime objects
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # 2. Sort by date so "past" really means "past"
    df = df.sort_values('Date')
    
    return df

def create_team_dataframe(df):
    """
    Reshapes match data so every row is a team's performance.
    Currently: 1 row = 1 match (Home vs Away)
    Target: 2 rows = 1 match (Row A: Home stats, Row B: Away stats)
    """
    
    # Create a list for the home team perspective
    home_df = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR']].copy()
    home_df.columns = ['Date', 'Team', 'GoalsScored', 'GoalsConceded', 'Result']
    home_df['IsHome'] = 1
    # If Result was 'H' (Home Win), it's a Win (3 pts). 'D' = 1, 'A' = 0
    home_df['Points'] = home_df['Result'].map({'H': 3, 'D': 1, 'A': 0})

    # Create a list for the away team perspective
    away_df = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR']].copy()
    away_df.columns = ['Date', 'Team', 'GoalsScored', 'GoalsConceded', 'Result']
    away_df['IsHome'] = 0
    # If Result was 'H' (Home Win), Away team lost (0 pts).
    away_df['Points'] = away_df['Result'].map({'H': 0, 'D': 1, 'A': 3})

    # Combine them
    team_df = pd.concat([home_df, away_df]).sort_values('Date')
    return team_df

def calculate_rolling_stats(team_df, window=5):
    """
    Calculates the average stats over the last 'window' games.
    IMPORTANT: We use .shift() so the current game's result isn't included in the average.
    """
    # Group by team so Arsenal stats don't mix with Chelsea stats
    grouped = team_df.groupby('Team')
    
    # Calculate rolling averages (closed='left' excludes current row in some pandas versions, 
    # but shift(1) is the safest way to prevent data leakage)
    
    team_df['Form_GoalsScored'] = grouped['GoalsScored'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
    team_df['Form_GoalsConceded'] = grouped['GoalsConceded'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
    team_df['Form_Points'] = grouped['Points'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
    
    # Fill N/A values (the first few games have no history) with 0 or averages
    team_df = team_df.fillna(0)
    
    return team_df

def merge_features_back(original_df, team_stats_df):
    """
    Takes the calculated stats and merges them back into the main Match fixture list.
    We need to attach 'Home Team Form' and 'Away Team Form'.
    """
    
    # 1. Attach Home Team Stats
    original_df = original_df.merge(
        team_stats_df[['Date', 'Team', 'Form_GoalsScored', 'Form_GoalsConceded', 'Form_Points']], 
        left_on=['Date', 'HomeTeam'], 
        right_on=['Date', 'Team'],
        how='left'
    )
    # Rename columns to be clear
    original_df = original_df.rename(columns={
        'Form_GoalsScored': 'Home_Goals_Avg',
        'Form_GoalsConceded': 'Home_Conceded_Avg', 
        'Form_Points': 'Home_Points_Avg'
    })
    original_df = original_df.drop(columns=['Team']) # cleanup

    # 2. Attach Away Team Stats
    original_df = original_df.merge(
        team_stats_df[['Date', 'Team', 'Form_GoalsScored', 'Form_GoalsConceded', 'Form_Points']], 
        left_on=['Date', 'AwayTeam'], 
        right_on=['Date', 'Team'],
        how='left'
    )
    original_df = original_df.rename(columns={
        'Form_GoalsScored': 'Away_Goals_Avg',
        'Form_GoalsConceded': 'Away_Conceded_Avg', 
        'Form_Points': 'Away_Points_Avg'
    })
    original_df = original_df.drop(columns=['Team']) # cleanup
    
    return original_df

if __name__ == "__main__":
    print("Processing data...")
    
    # 1. Load
    raw_df = load_and_clean_data()
    
    # 2. Reshape to Team-Centric view
    team_stats = create_team_dataframe(raw_df)
    
    # 3. Calculate "Form" (Last 5 games)
    team_stats_rolled = calculate_rolling_stats(team_stats, window=5)
    
    # 4. Merge back to Match view
    final_df = merge_features_back(raw_df, team_stats_rolled)
    
    # 5. Save
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    final_df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(f"Features engineered! Saved to {PROCESSED_DATA_PATH}")
    print("   New columns added: Home_Points_Avg, Away_Points_Avg, etc.")