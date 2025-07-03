import pandas as pd
import json
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import os
import pybaseball as pb
import unicodedata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PitcherDataCollector:
    """
    Collects pitcher statistics using pybaseball for ML training.
    
    Sources:
    - Baseball Reference (via pybaseball)
    - Fangraphs (via pybaseball)
    - Statcast (via pybaseball)
    - Custom scraped data
    """
    
    def __init__(self):
        """Initialize the data collector."""
        logger.info("Initializing PitcherDataCollector with pybaseball")
        
    def clean_player_name(self, name: str) -> str:
        """
        Clean player names by handling encoding issues and special characters.
        
        Args:
            name (str): Raw player name
            
        Returns:
            str: Cleaned player name
        """
        if pd.isna(name) or name is None:
            return ""
        
        # Convert to string if not already
        name = str(name)
        
        # Handle specific encoding issues we're seeing
        replacements = {
            '\\xc3\\xa1': 'a',  # á
            '\\xc3\\xa9': 'e',  # é
            '\\xc3\\xad': 'i',  # í
            '\\xc3\\xb3': 'o',  # ó
            '\\xc3\\xba': 'u',  # ú
            '\\xc3\\xb1': 'n',  # ñ
            '\\xc3\\xa7': 'c',  # ç
            '\\xc3\\xad': 'i',  # í
            '\\xc3\\xb3': 'o',  # ó
            '\\xc3\\xba': 'u',  # ú
            '\\xc3\\xb1': 'n',  # ñ
            '\\xc3\\xa7': 'c',  # ç
        }
        
        # Apply replacements
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        # Handle actual unicode characters (not escaped)
        try:
            # Try to decode if it's bytes
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='ignore')
            
            # Normalize unicode characters
            name = unicodedata.normalize('NFKD', name)
            
            # Remove or replace problematic characters
            name = name.replace('á', 'a')
            name = name.replace('é', 'e')
            name = name.replace('í', 'i')
            name = name.replace('ó', 'o')
            name = name.replace('ú', 'u')
            name = name.replace('ñ', 'n')
            name = name.replace('ç', 'c')
            
        except Exception:
            pass
        
        # Remove any remaining non-ASCII characters
        name = ''.join(char for char in name if ord(char) < 128)
        
        # Clean up extra whitespace
        name = ' '.join(name.split())
        
        return name
    
    def get_pitcher_stats(self, season: int = 2024) -> pd.DataFrame:
        """
        Get pitcher statistics using pybaseball.
        
        Args:
            season (int): MLB season year
            
        Returns:
            pd.DataFrame: Pitcher statistics
        """
        try:
            logger.info(f"Fetching pitcher stats for {season} season using pybaseball...")
            
            # Try the more comprehensive data source first (returns 855 pitchers)
            stats = pb.pitching_stats_bref(season)
            
            if stats is None or stats.empty:
                logger.warning(f"No data returned for {season} season")
                return pd.DataFrame()
            
            # Clean and standardize column names
            stats.columns = stats.columns.str.lower().str.replace(' ', '_')
            
            # Check what columns are actually available
            logger.info(f"Available columns: {list(stats.columns)}")
            
            # Select relevant columns that exist and rename for consistency
            available_columns = ['name', 'tm', 'age', 'w', 'l', 'era', 'g', 'gs', 'sv', 'ip', 
                               'h', 'r', 'er', 'hr', 'bb', 'so', 'whip']
            
            # Add per-9 columns if they exist (they might have different names)
            per9_columns = ['h9', 'hr9', 'bb9', 'so9', 'h/9', 'hr/9', 'bb/9', 'so/9', 'so9']
            for col in per9_columns:
                if col in stats.columns:
                    available_columns.append(col)
            
            # Filter to only include columns that exist
            existing_columns = [col for col in available_columns if col in stats.columns]
            pitcher_stats = stats[existing_columns].copy()
            
            # Rename columns to match our expected format
            column_mapping = {
                'tm': 'team',
                'w': 'wins',
                'l': 'losses',
                'g': 'games',
                'gs': 'games_started',
                'sv': 'saves',
                'ip': 'innings_pitched',
                'h': 'hits_allowed',
                'r': 'runs_allowed',
                'er': 'earned_runs',
                'hr': 'home_runs_allowed',
                'bb': 'walks',
                'so': 'strikeouts',
                'h9': 'hits_per_9',
                'hr9': 'hr_per_9',
                'bb9': 'bb_per_9',
                'so9': 'k_per_9',
                'h/9': 'hits_per_9',
                'hr/9': 'hr_per_9',
                'bb/9': 'bb_per_9',
                'so/9': 'k_per_9'
            }
            
            # Only rename columns that exist
            existing_mapping = {k: v for k, v in column_mapping.items() if k in pitcher_stats.columns}
            pitcher_stats = pitcher_stats.rename(columns=existing_mapping)
            
            # Add season column
            pitcher_stats['season'] = season
            
            # Clean player names
            pitcher_stats['name'] = pitcher_stats['name'].apply(self.clean_player_name)
            
            # Generate pitcher_id (simple hash of name for now)
            pitcher_stats['pitcher_id'] = pitcher_stats['name'].apply(lambda x: hash(str(x)) % 1000000)
            
            logger.info(f"Collected data for {len(pitcher_stats)} pitchers")
            return pitcher_stats
            
        except Exception as e:
            logger.error(f"Error collecting pitcher data: {e}")
            return pd.DataFrame()
    
    def get_pitcher_advanced_stats(self, season: int = 2024) -> pd.DataFrame:
        """
        Get advanced pitcher statistics from Fangraphs.
        
        Args:
            season (int): MLB season year
            
        Returns:
            pd.DataFrame: Advanced pitcher statistics
        """
        try:
            logger.info(f"Fetching advanced pitcher stats for {season} season...")
            
            # Get advanced stats from Fangraphs (different source than basic stats)
            advanced_stats = pb.pitching_stats(season)
            
            if advanced_stats is None or advanced_stats.empty:
                logger.warning(f"No advanced data returned for {season} season")
                return pd.DataFrame()
            
            # Clean column names
            advanced_stats.columns = advanced_stats.columns.str.lower().str.replace(' ', '_')
            
            # Check what columns are actually available
            logger.info(f"Available advanced columns: {list(advanced_stats.columns)}")
            
            # Select relevant advanced metrics
            relevant_columns = [
                'name', 'team', 'era', 'fip', 'xfip', 'whip', 'k/9', 'bb/9', 'hr/9',
                'k%', 'bb%', 'hr%', 'babip', 'lob%', 'gb%', 'fb%', 'hr/fb'
            ]
            
            available_columns = [col for col in relevant_columns if col in advanced_stats.columns]
            advanced_pitcher_stats = advanced_stats[available_columns].copy()
            
            # Rename columns for consistency
            column_mapping = {
                'k/9': 'k_per_9',
                'bb/9': 'bb_per_9', 
                'hr/9': 'hr_per_9',
                'k%': 'k_percent',
                'bb%': 'bb_percent',
                'hr%': 'hr_percent',
                'lob%': 'lob_percent',
                'gb%': 'gb_percent',
                'fb%': 'fb_percent',
                'hr/fb': 'hr_fb_ratio'
            }
            
            # Only rename columns that exist
            existing_mapping = {k: v for k, v in column_mapping.items() if k in advanced_pitcher_stats.columns}
            advanced_pitcher_stats = advanced_pitcher_stats.rename(columns=existing_mapping)
            
            # Add season column
            advanced_pitcher_stats['season'] = season
            
            # Clean player names
            advanced_pitcher_stats['name'] = advanced_pitcher_stats['name'].apply(self.clean_player_name)
            
            logger.info(f"Collected advanced data for {len(advanced_pitcher_stats)} pitchers")
            return advanced_pitcher_stats
            
        except Exception as e:
            logger.error(f"Error collecting advanced pitcher data: {e}")
            return pd.DataFrame()
    
    def get_statcast_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get Statcast data for all pitchers in a date range.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Statcast pitcher data
        """
        try:
            logger.info(f"Fetching Statcast data from {start_date} to {end_date}...")
            
            # Get Statcast data for all pitchers (not specific player)
            statcast_data = pb.statcast(start_date, end_date)
            
            if statcast_data is None or statcast_data.empty:
                logger.warning(f"No Statcast data returned for date range")
                return pd.DataFrame()
            
            # Filter for pitching events only
            pitching_data = statcast_data[statcast_data['pitch_type'].notna()].copy()
            
            if pitching_data.empty:
                logger.warning("No pitching data found in Statcast")
                return pd.DataFrame()
            
            # Check what columns are actually available
            logger.info(f"Available Statcast columns: {list(pitching_data.columns)}")
            
            # Define aggregation functions based on available columns
            agg_functions = {
                'pitch_type': 'count',  # Total pitches
                'game_date': 'nunique'  # Unique games pitched
            }
            
            # Add velocity stats if available
            if 'release_speed' in pitching_data.columns:
                agg_functions['release_speed'] = ['mean', 'max']
            
            # Add spin rate stats if available
            if 'release_spin_rate' in pitching_data.columns:
                agg_functions['release_spin_rate'] = 'mean'
            
            # Add pitch name if available
            if 'pitch_name' in pitching_data.columns:
                agg_functions['pitch_name'] = lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
            
            # Add zone if available
            if 'zone' in pitching_data.columns:
                agg_functions['zone'] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
            
            # Add throwing hand if available
            if 'p_throws' in pitching_data.columns:
                agg_functions['p_throws'] = 'first'
            
            # Add batter stance if available
            if 'stand' in pitching_data.columns:
                agg_functions['stand'] = lambda x: x.value_counts().index[0] if len(x) > 0 else 'R'
            
            # Add ball/strike counts if available
            if 'balls' in pitching_data.columns:
                agg_functions['balls'] = 'sum'
            if 'strikes' in pitching_data.columns:
                agg_functions['strikes'] = 'sum'
            
            # Aggregate by pitcher
            pitcher_aggregates = pitching_data.groupby('pitcher').agg(agg_functions).reset_index()
            
            # Flatten column names
            if len(pitcher_aggregates.columns) > 2:  # More than just pitcher and one metric
                pitcher_aggregates.columns = [
                    'pitcher_id', 'total_pitches', 'games_pitched'
                ] + [f'stat_{i}' for i in range(len(pitcher_aggregates.columns) - 3)]
            else:
                pitcher_aggregates.columns = ['pitcher_id', 'total_pitches']
            
            # Add pitcher names (you might need to get this from another source)
            pitcher_aggregates['pitcher_name'] = f"Pitcher_{pitcher_aggregates['pitcher_id']}"
            
            logger.info(f"Collected Statcast data for {len(pitcher_aggregates)} pitchers")
            return pitcher_aggregates
            
        except Exception as e:
            logger.error(f"Error collecting Statcast data: {e}")
            return pd.DataFrame()
    
    def create_training_dataset(self, seasons: List[int] = None) -> None:
        """
        Create a training dataset for each season and save as separate CSV files.
        Args:
            seasons (List[int]): List of MLB season years to collect data for.
                                Defaults to [2021, 2022, 2023, 2024, 2025]
        """
        if seasons is None:
            seasons = [2021, 2022, 2023, 2024, 2025]
        
        summary = {}
        try:
            logger.info(f"Creating separate training datasets for seasons: {seasons}")
            os.makedirs('data', exist_ok=True)
            for season in seasons:
                logger.info(f"Processing season {season}...")
                # Get basic season stats
                season_stats = self.get_pitcher_stats(season)
                if season_stats.empty:
                    logger.warning(f"No data available for {season} season, skipping...")
                    continue
                # Get advanced stats for this season
                advanced_stats = self.get_pitcher_advanced_stats(season)
                # Get previous season stats for comparison (if available)
                prev_season = season - 1
                prev_season_stats = None
                if prev_season >= min(seasons):
                    prev_season_stats = self.get_pitcher_stats(prev_season)
                # Start with basic season stats
                training_data = season_stats.copy()
                # Merge with advanced stats (only if team column exists in both)
                if not advanced_stats.empty and 'team' in advanced_stats.columns:
                    basic_cols = set(training_data.columns)
                    advanced_cols = set(advanced_stats.columns)
                    common_cols = basic_cols.intersection(advanced_cols)
                    advanced_stats_clean = advanced_stats.drop(columns=list(common_cols - {'name', 'team', 'season'}))
                    training_data = training_data.merge(
                        advanced_stats_clean, 
                        on=['name', 'team', 'season'], 
                        how='left'
                    )
                elif not advanced_stats.empty:
                    basic_cols = set(training_data.columns)
                    advanced_cols = set(advanced_stats.columns)
                    common_cols = basic_cols.intersection(advanced_cols)
                    advanced_stats_clean = advanced_stats.drop(columns=list(common_cols - {'name', 'season'}))
                    training_data = training_data.merge(
                        advanced_stats_clean, 
                        on=['name', 'season'], 
                        how='left'
                    )
                # Merge with previous season stats
                if prev_season_stats is not None and not prev_season_stats.empty:
                    merge_columns = ['name']
                    if 'team' in prev_season_stats.columns:
                        merge_columns.append('team')
                    prev_stats_subset = prev_season_stats[merge_columns + ['era', 'whip']].copy()
                    prev_stats_subset = prev_stats_subset.rename(columns={
                        'era': 'prev_era',
                        'whip': 'prev_whip'
                    })
                    training_data = training_data.merge(
                        prev_stats_subset,
                        on=merge_columns,
                        how='left'
                    )
                # Add derived features (only if required columns exist)
                if 'strikeouts' in training_data.columns and 'innings_pitched' in training_data.columns:
                    training_data['strikeout_rate'] = training_data['strikeouts'] / training_data['innings_pitched'].replace(0, 1)
                if 'walks' in training_data.columns and 'innings_pitched' in training_data.columns:
                    training_data['walk_rate'] = training_data['walks'] / training_data['innings_pitched'].replace(0, 1)
                if 'strikeouts' in training_data.columns and 'walks' in training_data.columns:
                    training_data['k_bb_ratio'] = training_data['strikeouts'] / training_data['walks'].replace(0, 1)
                if 'hits_allowed' in training_data.columns and 'innings_pitched' in training_data.columns:
                    training_data['hits_per_9'] = training_data['hits_allowed'] / training_data['innings_pitched'] * 9
                if 'runs_allowed' in training_data.columns and 'innings_pitched' in training_data.columns:
                    training_data['runs_per_9'] = training_data['runs_allowed'] / training_data['innings_pitched'] * 9
                # Add season-specific features
                training_data['season_year'] = season
                training_data['years_experience'] = season - training_data['age'].fillna(season - 18)
                # Fill missing values
                training_data = training_data.fillna(0)
                # Save to CSV for this season
                out_path = f"data/pitcher_training_data_{season}.csv"
                training_data.to_csv(out_path, index=False)
                logger.info(f"Saved {len(training_data)} records for {season} season to {out_path}")
                summary[season] = len(training_data)
            logger.info(f"Summary of records per season: {summary}")
            print("\nSummary of records per season:")
            for season, count in summary.items():
                print(f"  {season}: {count} records (data/pitcher_training_data_{season}.csv)")
        except Exception as e:
            logger.error(f"Error creating training datasets: {e}")
            print(f"Error creating training datasets: {e}")


if __name__ == "__main__":
    # Test the data collector
    collector = PitcherDataCollector()
    
    # Create training datasets for multiple seasons
    seasons = [2021, 2022, 2023, 2024, 2025]
    collector.create_training_dataset(seasons)
    print("\nDone! Check the data/ directory for per-season CSV files.") 