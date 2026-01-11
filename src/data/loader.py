import pandas as pd
import os

# 1. Configuration
# We save to 'data/raw' so we don't mix it with cleaned data later
DATA_DIR = os.path.join("data", "raw")

# URLs for English Premier League (E0) from 2019 to 2024
SEASONS = [
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv", # 2023-2024
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv", # 2022-2023
    "https://www.football-data.co.uk/mmz4281/2122/E0.csv", # 2021-2022
    "https://www.football-data.co.uk/mmz4281/2021/E0.csv", # 2020-2021
    "https://www.football-data.co.uk/mmz4281/1920/E0.csv", # 2019-2020
]

def download_data():
    """Downloads historical EPL data and saves it to data/raw."""
    
    # Ensure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    all_seasons = []
    
    print(f"Starting download for {len(SEASONS)} seasons...")
    
    for url in SEASONS:
        try:
            # Extract season name (e.g., '2324')
            season_id = url.split('/')[-2]
            print(f"   Downloading season {season_id}...")
            
            # Read directly from URL into Pandas
            df = pd.read_csv(url)
            
            # Tag the rows with the season ID so we don't lose track later
            df['season_id'] = season_id
            
            all_seasons.append(df)
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")

    # Combine all individual season tables into one big table
    if all_seasons:
        full_history = pd.concat(all_seasons, ignore_index=True)
        
        # Save to local CSV
        save_path = os.path.join(DATA_DIR, "epl_history_raw.csv")
        full_history.to_csv(save_path, index=False)
        
        print(f"\nSuccess! Saved {len(full_history)} matches to {save_path}")
        print(f"   Columns available include: Div, Date, HomeTeam, AwayTeam, FTHG (Home Goals), FTAG (Away Goals), FTR (Result)")
    else:
        print("\nNo data downloaded.")

if __name__ == "__main__":
    download_data()