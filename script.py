import requests
import json
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse

# URL to scrape
url = "https://sportsbook.draftkings.com/leagues/baseball/mlb?category=pitcher-props&subcategory=strikeouts-thrown-o%2Fu"

def parse_row_html(html_content):
    """
    Parse a single row HTML and extract structured data
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the table row
    row = soup.find('tr')
    if not row:
        return None
    
    # Extract player information
    player_data = {}
    
    # Get player name
    player_name_elem = row.find('span', class_='sportsbook-row-name')
    if player_name_elem:
        player_data['player_name'] = player_name_elem.get_text(strip=True)
    
    # Get player image URL
    player_img = row.find('img', class_='sportsbook-player-image')
    if player_img:
        player_data['player_image'] = player_img.get('src', '')
    
    # Get player stats (total)
    player_stats = row.find('div', class_='player-stats')
    if player_stats:
        # Extract the total value from the digit elements
        digit_elements = player_stats.find_all('div', class_='player-stats__digit', attrs={'style': lambda x: x and 'translateY(0%)' in x})
        if digit_elements:
            total_value = ''.join([elem.get_text(strip=True) for elem in digit_elements])
            player_data['total'] = total_value
    
    # Extract betting odds from all outcome cells
    outcome_cells = row.find_all('div', class_='sportsbook-outcome-cell')
    betting_data = []
    
    for cell in outcome_cells:
        cell_data = {}
        
        # Get label (O/U)
        label_elem = cell.find('span', class_='sportsbook-outcome-cell__label')
        if label_elem:
            cell_data['label'] = label_elem.get_text(strip=True)
        
        # Get line value
        line_elem = cell.find('span', class_='sportsbook-outcome-cell__line')
        if line_elem:
            cell_data['line'] = line_elem.get_text(strip=True)
        
        # Get odds
        odds_elem = cell.find('span', class_='sportsbook-odds')
        if odds_elem:
            cell_data['odds'] = odds_elem.get_text(strip=True)
        
        # Get aria-label for additional context
        button = cell.find('div', class_='sportsbook-outcome-cell__body')
        if button:
            cell_data['aria_label'] = button.get('aria-label', '')
        
        if cell_data:
            betting_data.append(cell_data)
    
    return {
        'player': player_data,
        'betting_options': betting_data
    }

def scrape_draftkings_data():
    """
    Scrape data from DraftKings sportsbook using BeautifulSoup with targeted selectors
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        print(f"Making request to: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        print("Parsing HTML with BeautifulSoup...")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try multiple selectors to find the table rows
        selectors = [
            "table.sportsbook-table tbody tr",
            ".sportsbook-table tbody tr",
            "table tbody tr",
            ".two-way-list-8 table tbody tr",
            "tr.sportsbook-table__row"
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
            
            # Save page source for debugging
            with open("page_source_css.html", "w", encoding="utf-8") as f:
                f.write(soup.prettify())
            print("Page source saved to page_source_css.html for debugging")
            
            return None
        
        all_rows_data = []
        
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
            
            # Get player image URL
            player_img = row.find('img', class_='sportsbook-player-image')
            if player_img:
                player_data['player_image'] = player_img.get('src', '')
            
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
            
            row_data = {
                "row_index": i + 1,
                "player": player_data,
                "betting_options": betting_data
            }
            
            all_rows_data.append(row_data)
        
        # Save data to JSON file
        scraped_data = {
            "url": url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_rows": len(rows),
            "selector_used": used_selector,
            "rows": all_rows_data
        }
        
        with open("scraped_data_css.json", "w") as f:
            json.dump(scraped_data, f, indent=2)
        
        print(f"\nData saved to scraped_data_css.json")
        return scraped_data
        
    except Exception as e:
        print(f"Error scraping data: {e}")
        return None

def test_parse_sample_html():
    """
    Test function to parse the sample HTML provided by user
    """
    sample_html = '''<tr class=""><th scope="row" class="sportsbook-table__column-row" style="width: unset;"><div class="player-stats-wrapper"><div class="sportsbook-row-name__wrapper" data-img="player-image"><a data-tracking="{&quot;target&quot;:&quot;PLAYER_PAGE_TRACK&quot;,&quot;action&quot;:&quot;click&quot;,&quot;section&quot;:&quot;LeaguePage&quot;,&quot;playerId&quot;:&quot;625767&quot;,&quot;player&quot;:&quot;Brandon Walter&quot;}" class="sportsbook-row-name__wrapper-link" href="/players/baseball/mlb/brandon-walter-odds-625767"><img class="sportsbook-player-image" src="blob:https://sportsbook.draftkings.com/8e4e8e40-128f-406b-921a-777ca1b81a97" alt="player image" width="36" height="36" data-testid="sportsbook-player-image"><span class="sportsbook-row-name" data-testid="row-name">Brandon Walter</span></a></div><div class="player-stats with-player-image" data-testid="player-stats">Total:&nbsp;<div class="player-stats__highlight"><div class="player-stats__digits-wrapper"><div class="player-stats__digit" style="transform: translateY(-300%);" data-testid="digit-0">0</div><div class="player-stats__digit" style="transform: translateY(-200%);" data-testid="digit-1">1</div><div class="player-stats__digit" style="transform: translateY(-100%);" data-testid="digit-2">2</div><div class="player-stats__digit active" style="transform: translateY(0%);" data-testid="digit-3">3</div><div class="player-stats__digit" style="transform: translateY(100%);" data-testid="digit-3">4</div><div class="player-stats__digit" style="transform: translateY(200%);" data-testid="digit-5">5</div><div class="player-stats__digit" style="transform: translateY(300%);" data-testid="digit-6">6</div><div class="player-stats__digit" style="transform: translateY(400%);" data-testid="digit-7">7</div><div class="player-stats__digit" style="transform: translateY(500%);" data-testid="digit-8">8</div><div class="player-stats__digit" style="transform: translateY(600%);" data-testid="digit-9">9</div></div><div class="player-stats__digits-wrapper"><div class="player-stats__digit active" style="transform: translateY(0%);" data-testid="digit-0">0</div><div class="player-stats__digit" style="transform: translateY(100%);" data-testid="digit-1">1</div><div class="player-stats__digit" style="transform: translateY(200%);" data-testid="digit-2">2</div><div class="player-stats__digit" style="transform: translateY(300%);" data-testid="digit-3">3</div><div class="player-stats__digit" style="transform: translateY(400%);" data-testid="digit-4">4</div><div class="player-stats__digit" style="transform: translateY(500%);" data-testid="digit-5">5</div><div class="player-stats__digit" style="transform: translateY(600%);" data-testid="digit-6">6</div><div class="player-stats__digit" style="transform: translateY(700%);" data-testid="digit-7">7</div><div class="player-stats__digit" style="transform: translateY(800%);" data-testid="digit-8">8</div><div class="player-stats__digit" style="transform: translateY(900%);" data-testid="digit-9">9</div></div></div></div></div></th><td class="sportsbook-table__column-row" style="width: unset;"><div class="sportsbook-outcome-cell" data-testid="sportsbook-outcome-cell"><div role="button" tabindex="0" aria-pressed="false" class="sportsbook-outcome-cell__body" data-testid="sportsbook-outcome-cell-button" aria-label="O 4.5" data-tracking="{&quot;section&quot;:&quot;GamesComponent&quot;,&quot;action&quot;:&quot;click&quot;,&quot;target&quot;:&quot;RemoveBet&quot;,&quot;sportName&quot;:&quot;7&quot;,&quot;leagueName&quot;:&quot;84240&quot;,&quot;subcategoryId&quot;:15221,&quot;eventId&quot;:&quot;32483818&quot;}"><div class="sportsbook-outcome-body-wrapper"><div class="sportsbook-outcome-cell__label-line-container"><span class="sportsbook-outcome-cell__label">O</span><span>&nbsp;</span><span class="sportsbook-outcome-cell__line" data-testid="sportsbook-outcome-cell-line">4.5</span></div><div class="sportsbook-outcome-cell__elements"><div class="sportsbook-outcome-cell__element" data-testid="sportsbook-outcome-cell-boost"></div><div class="sportsbook-outcome-cell__element" data-testid="sportsbook-outcome-cell-element"><span data-testid="sportsbook-odds" class="sportsbook-odds american default-color">+105</span></div></div></div></div></div></td><td class="sportsbook-table__column-row" style="width: unset;"><div class="sportsbook-outcome-cell" data-testid="sportsbook-outcome-cell"><div role="button" tabindex="0" aria-pressed="false" class="sportsbook-outcome-cell__body" data-testid="sportsbook-outcome-cell-button" aria-label="U 4.5" data-tracking="{&quot;section&quot;:&quot;GamesComponent&quot;,&quot;action&quot;:&quot;click&quot;,&quot;target&quot;:&quot;RemoveBet&quot;,&quot;sportName&quot;:&quot;7&quot;,&quot;leagueName&quot;:&quot;84240&quot;,&quot;subcategoryId&quot;:15221,&quot;eventId&quot;:&quot;32483818&quot;}"><div class="sportsbook-outcome-body-wrapper"><div class="sportsbook-outcome-cell__label-line-container"><span class="sportsbook-outcome-cell__label">U</span><span>&nbsp;</span><span class="sportsbook-outcome-cell__line" data-testid="sportsbook-outcome-cell-line">4.5</span></div><div class="sportsbook-outcome-cell__elements"><div class="sportsbook-outcome-cell__element" data-testid="sportsbook-outcome-cell-boost"></div><div class="sportsbook-outcome-cell__element" data-testid="sportsbook-outcome-cell-element"><span data-testid="sportsbook-odds" class="sportsbook-odds american default-color">−140</span></div></div></div></div></div></td></tr>'''
    
    print("Testing parse with sample HTML...")
    result = parse_row_html(sample_html)
    
    if result:
        print("Parsed data:")
        print(json.dumps(result, indent=2))
        
        # Save test result
        with open("test_parse_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Test result saved to test_parse_result.json")
    else:
        print("Failed to parse sample HTML")

def scrape_with_requests():
    """
    Alternative method using requests and BeautifulSoup (may not work due to JavaScript)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find the table using a more flexible approach
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables on the page")
        
        for i, table in enumerate(tables):
            print(f"Table {i + 1}:")
            rows = table.find_all('tr')
            for j, row in enumerate(rows[:3]):  # Show first 3 rows
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                print(f"  Row {j + 1}: {row_data}")
        
        return soup
        
    except Exception as e:
        print(f"Error with requests method: {e}")
        return None

if __name__ == "__main__":
    print("Starting DraftKings data scraping with BeautifulSoup...")
    
    # First test with the sample HTML
    print("\n=== Testing with sample HTML ===")
    test_parse_sample_html()
    
    # Then try scraping the live site
    print("\n=== Scraping live site ===")
    result = scrape_draftkings_data()
    
    if not result:
        print("\n=== Trying alternative requests method ===")
        scrape_with_requests()
    
    print("\nScraping completed!")