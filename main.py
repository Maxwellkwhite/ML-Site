from flask import Flask, render_template, jsonify, request
from flask_bootstrap import Bootstrap5
import os
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

app = Flask(__name__)

# Initialize Bootstrap
bootstrap = Bootstrap5(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

# Data file paths
STRIKEOUT_PROPS_FILE = 'strikeout_props.json'

def load_cached_data(filename, max_age_hours=1):
    """Load data from JSON file if it exists and is not too old"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Check if data is fresh enough
            if 'timestamp' in data:
                timestamp = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - timestamp < timedelta(hours=max_age_hours):
                    print(f"Using cached data from {filename}")
                    return data
                else:
                    print(f"Cached data in {filename} is too old")
            else:
                print(f"No timestamp found in {filename}")
        else:
            print(f"No cached data file found: {filename}")
    except Exception as e:
        print(f"Error loading cached data from {filename}: {e}")
    
    return None

def save_data_to_cache(data, filename):
    """Save data to JSON file with timestamp"""
    try:
        data_with_timestamp = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        with open(filename, 'w') as f:
            json.dump(data_with_timestamp, f, indent=2)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")

def debug_draftkings_structure():
    """Debug function to inspect DraftKings HTML structure"""
    try:
        url = "https://sportsbook.draftkings.com/leagues/baseball/mlb?category=pitcher-props&subcategory=strikeouts-thrown-o%2Fu"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print("=== DRAFTKINGS HTML STRUCTURE ANALYSIS ===")
        
        # Look for common patterns in sports betting sites
        print("\n1. Looking for pitcher names...")
        pitcher_elements = soup.find_all(['div', 'span', 'a'], string=lambda text: text and any(name in text for name in ['Festa', 'Pérez', 'Walter', 'Freeland']))
        print(f"Found {len(pitcher_elements)} potential pitcher elements")
        for elem in pitcher_elements[:5]:  # Show first 5
            print(f"  - {elem.name}: {elem.get_text(strip=True)[:50]}...")
        
        print("\n2. Looking for strikeout-related elements...")
        strikeout_elements = soup.find_all(['div', 'span'], string=lambda text: text and ('strikeout' in text.lower() or 'k' in text.lower() or 'so' in text.lower()))
        print(f"Found {len(strikeout_elements)} potential strikeout elements")
        for elem in strikeout_elements[:5]:  # Show first 5
            print(f"  - {elem.name}: {elem.get_text(strip=True)[:50]}...")
        
        print("\n3. Looking for over/under elements...")
        ou_elements = soup.find_all(['div', 'span'], string=lambda text: text and ('over' in text.lower() or 'under' in text.lower() or 'o/u' in text.lower()))
        print(f"Found {len(ou_elements)} potential over/under elements")
        for elem in ou_elements[:5]:  # Show first 5
            print(f"  - {elem.name}: {elem.get_text(strip=True)[:50]}...")
        
        print("\n4. Looking for numeric values (potential odds)...")
        numeric_elements = soup.find_all(['div', 'span'], string=lambda text: text and any(char.isdigit() for char in text) and '.' in text)
        print(f"Found {len(numeric_elements)} potential numeric elements")
        for elem in numeric_elements[:10]:  # Show first 10
            text = elem.get_text(strip=True)
            if len(text) < 20:  # Only show short ones
                print(f"  - {elem.name}: {text}")
        
        print("\n5. Looking for common class patterns...")
        all_classes = set()
        for tag in soup.find_all(class_=True):
            all_classes.update(tag.get('class', []))
        
        relevant_classes = [cls for cls in all_classes if any(keyword in cls.lower() for keyword in ['pitcher', 'strikeout', 'prop', 'odds', 'player', 'name'])]
        print(f"Found {len(relevant_classes)} relevant classes:")
        for cls in relevant_classes[:10]:  # Show first 10
            print(f"  - {cls}")
        
        # Save HTML to file for manual inspection
        with open('draftkings_debug.html', 'w', encoding='utf-8') as f:
            f.write(str(soup))
        print("\n6. Saved full HTML to 'draftkings_debug.html' for manual inspection")
        
        return soup
        
    except Exception as e:
        print(f"Error in debug function: {e}")
        return None

def scrape_draftkings_strikeout_props():
    """Scrape strikeout over/under props from DraftKings using Selenium"""
    try:
        url = "https://sportsbook.draftkings.com/leagues/baseball/mlb?category=pitcher-props&subcategory=strikeouts-thrown-o%2Fu"
        
        # Set up Chrome options for headless browsing
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Add user agent to avoid detection
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        driver = None
        try:
            # Initialize the driver
            driver = webdriver.Chrome(options=chrome_options)
            
            print(f"Navigating to: {url}")
            driver.get(url)
            
            # Wait for the page to load
            wait = WebDriverWait(driver, 30)
            
            print("Waiting for page to load...")
            
            # Wait for any table to be present
            try:
                # Wait for table elements to load
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
                print("Found table elements!")
                
                # Additional wait for dynamic content
                time.sleep(5)
                
            except Exception as e:
                print(f"Timeout waiting for table: {e}")
            
            # Get the page source after JavaScript has rendered
            page_source = driver.page_source
            print("Got page source, parsing with BeautifulSoup...")
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Try multiple selectors to find the table rows
            selectors = [
                "table.sportsbook-table tbody tr",
                ".sportsbook-table tbody tr",
                "table tbody tr",
                ".two-way-list-8 table tbody tr",
                "tr.sportsbook-table__row",
                "tr"
            ]
            
            rows = []
            used_selector = None
            
            for selector in selectors:
                print(f"Trying selector: {selector}")
                rows = soup.select(selector)
                if rows:
                    used_selector = selector
                    print(f"Found {len(rows)} rows with selector: {selector}")
                    break
            
            if not rows:
                print("No table rows found with any selector")
                return {}, []
            
            # Dictionary to store pitcher strikeout props
            strikeout_props = {}
            all_pitchers = []
            
            for i, row in enumerate(rows):
                print(f"\nProcessing row {i + 1}:")
                
                # Extract player information
                player_data = {}
                
                # Get player name
                player_name_elem = row.find('span', class_='sportsbook-row-name')
                if player_name_elem:
                    player_name = player_name_elem.get_text(strip=True)
                    player_data['player_name'] = player_name
                    print(f"  Player: {player_name}")
                    
                    # Add to all pitchers list
                    all_pitchers.append(player_name)
                
                # Get player stats (total)
                player_stats = row.find('div', class_='player-stats')
                if player_stats:
                    # Extract the total value from the digit elements
                    digit_elements = player_stats.find_all('div', class_='player-stats__digit', attrs={'style': lambda x: x and 'translateY(0%)' in x})
                    if digit_elements:
                        total_value = ''.join([elem.get_text(strip=True) for elem in digit_elements])
                        player_data['total'] = total_value
                        print(f"  Total: {total_value}")
                
                # Extract betting odds from all outcome cells
                outcome_cells = row.find_all('div', class_='sportsbook-outcome-cell')
                betting_data = []
                
                for j, cell in enumerate(outcome_cells):
                    cell_data = {}
                    
                    # Get label (O/U)
                    label_elem = cell.find('span', class_='sportsbook-outcome-cell__label')
                    if label_elem:
                        label = label_elem.get_text(strip=True)
                        cell_data['label'] = label
                    
                    # Get line value
                    line_elem = cell.find('span', class_='sportsbook-outcome-cell__line')
                    if line_elem:
                        line = line_elem.get_text(strip=True)
                        cell_data['line'] = line
                    
                    # Get odds
                    odds_elem = cell.find('span', class_='sportsbook-odds')
                    if odds_elem:
                        odds = odds_elem.get_text(strip=True)
                        cell_data['odds'] = odds
                    
                    # Get aria-label for additional context
                    button = cell.find('div', class_='sportsbook-outcome-cell__body')
                    if button:
                        cell_data['aria_label'] = button.get('aria-label', '')
                    
                    if cell_data:
                        betting_data.append(cell_data)
                        print(f"  Betting option {j + 1}: {cell_data}")
                
                # Store the strikeout line for this pitcher
                if player_data.get('player_name') and betting_data:
                    # Look for the "Over" line (first betting option)
                    for bet in betting_data:
                        if bet.get('label') == 'O' and bet.get('line'):
                            strikeout_props[player_data['player_name']] = bet['line']
                            print(f"Found strikeout prop: {player_data['player_name']} - O/U {bet['line']}")
                            break
            
            print(f"Successfully found {len(strikeout_props)} strikeout props and {len(all_pitchers)} total pitchers")
            return strikeout_props, all_pitchers
            
        except Exception as e:
            print(f"Error scraping data: {e}")
            return {}, []
            
        finally:
            try:
                if driver:
                    driver.quit()
            except:
                pass
        
    except Exception as e:
        print(f"Error in scrape_draftkings_strikeout_props: {e}")
        return {}, []

def format_draftkings_pitchers_data(strikeout_props, all_pitchers):
    """Format DraftKings data in the same structure as MLB data"""
    games = []
    
    # Group pitchers into pairs (every 2 pitchers make a game)
    for i in range(0, len(all_pitchers), 2):
        if i + 1 < len(all_pitchers):
            away_pitcher = all_pitchers[i]
            home_pitcher = all_pitchers[i + 1]
            
            games.append({
                'away_team': 'TBD',  # DraftKings doesn't provide team info in this context
                'home_team': 'TBD',
                'away_pitcher': away_pitcher,
                'home_pitcher': home_pitcher,
                'game_time': 'TBD'
            })
            print(f"Added game from DraftKings: {away_pitcher} vs {home_pitcher}")
        else:
            # Handle odd number of pitchers (last pitcher gets a solo game)
            away_pitcher = all_pitchers[i]
            games.append({
                'away_team': 'TBD',
                'home_team': 'TBD',
                'away_pitcher': away_pitcher,
                'home_pitcher': 'TBD',
                'game_time': 'TBD'
            })
            print(f"Added solo game from DraftKings: {away_pitcher}")
    
    print(f"Total games formatted from DraftKings: {len(games)}")
    return games



@app.route('/debug-draftkings')
def debug_draftkings():
    """Debug route to analyze DraftKings HTML structure"""
    debug_draftkings_structure()
    return "Debug analysis complete. Check console output and draftkings_debug.html file."

@app.route('/refresh-data')
def refresh_data():
    """Manual refresh endpoint to update cached data"""
    try:
        print("Manual refresh requested...")
        
        # Scrape fresh data from DraftKings
        strikeout_props, all_pitchers = scrape_draftkings_strikeout_props()
        
        # Save both strikeout props and pitcher list to the same file
        cache_data = {
            'strikeout_props': strikeout_props,
            'all_pitchers': all_pitchers
        }
        save_data_to_cache(cache_data, STRIKEOUT_PROPS_FILE)
        
        return jsonify({
            'success': True,
            'message': f'Data refreshed successfully. Found {len(all_pitchers)} pitchers and {len(strikeout_props)} strikeout props.',
            'pitchers_count': len(all_pitchers),
            'strikeout_props_count': len(strikeout_props)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error refreshing data: {str(e)}'
        }), 500

@app.route('/', methods=["GET", "POST"])
def home_page():
    year = datetime.now().year
    
    # Try to load cached data first
    cached_strikeout_props = load_cached_data(STRIKEOUT_PROPS_FILE, max_age_hours=1)
    
    # If no cached data or data is stale, scrape fresh data
    if not cached_strikeout_props:
        print("No cached data, scraping fresh data...")
        strikeout_props, all_pitchers = scrape_draftkings_strikeout_props()
        # Save both strikeout props and pitcher list to the same file
        cache_data = {
            'strikeout_props': strikeout_props,
            'all_pitchers': all_pitchers
        }
        save_data_to_cache(cache_data, STRIKEOUT_PROPS_FILE)
    else:
        print("Using cached data...")
        strikeout_props = cached_strikeout_props['data']['strikeout_props']
        all_pitchers = cached_strikeout_props['data']['all_pitchers']
    
    # Format the DraftKings data in the same structure as MLB data
    starting_pitchers = format_draftkings_pitchers_data(strikeout_props, all_pitchers)
    
    # Add strikeout props to each pitcher
    for game in starting_pitchers:
        if game['away_pitcher'] in strikeout_props:
            game['away_strikeouts'] = strikeout_props[game['away_pitcher']]
        else:
            game['away_strikeouts'] = 'Not Available'
            
        if game['home_pitcher'] in strikeout_props:
            game['home_strikeouts'] = strikeout_props[game['home_pitcher']]
        else:
            game['home_strikeouts'] = 'Not Available'
    
    return render_template('index.html', year=year, starting_pitchers=starting_pitchers, datetime=datetime)

if __name__ == "__main__":
    app.run(debug=True, port=5002)




