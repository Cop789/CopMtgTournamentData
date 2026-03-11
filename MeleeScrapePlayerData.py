from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

# ---- CONFIG ----
TournamentNumber = "408838"  # Replace with actual tournament number
URL = f"https://melee.gg/Tournament/View/{TournamentNumber}"

# Set your desired folder and filename here
save_folder = r"Data\PlayerData"
csv_filename = f"playerData_{TournamentNumber}.csv"

# Ensure folder exists
os.makedirs(save_folder, exist_ok=True)
csv_path = os.path.join(save_folder, csv_filename)

# ---- SETUP DRIVER ----
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

# ---- OPEN PAGE ----
driver.get(URL)

# ---- HANDLE COOKIE POPUP ----
try:
    necessary_button = wait.until(
        EC.element_to_be_clickable((By.ID, "necessaryOnlyButton"))
    )
    necessary_button.click()
except:
    print("Cookie popup not found, continuing...")

# ---- SCRAPE TABLE ----
results = []

while True:
    # Wait for table to be present
    wait.until(EC.presence_of_element_located((By.ID, "tournament-standings-table")))

    # Get current page HTML and parse
    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("table", id="tournament-standings-table")
    
    # Extract rows
    for row in table.find("tbody").find_all("tr"):
        player = row.find("a", {"data-type": "player"})
        deck = row.find("a", {"data-type": "decklist"})
        rank = row.find("td", class_="Rank-column sorting_1")
        if player and deck and rank:
            results.append([player.text.strip(), deck.text.strip(), rank.text.strip()])

    # Wait for NEXT button to be present
    try:
        next_button = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, "tournament-standings-table_next"))
        )
    except:
        print("Next button not found, stopping pagination.")
        break

    # Stop if NEXT is disabled
    if "disabled" in next_button.get_attribute("class"):
        break

    # Get current page number
    current_page = int(driver.find_element(
        By.CSS_SELECTOR, "#tournament-standings-table_paginate a.current"
    ).text)

    # Click NEXT
    next_button.click()

    # Wait for page number to change
    wait.until(
        lambda d: int(d.find_element(By.CSS_SELECTOR, "#tournament-standings-table_paginate a.current").text) != current_page
    )

    time.sleep(1)  # small delay to avoid issues

# ---- SAVE TO CSV ----
df = pd.DataFrame(results, columns=["Player", "Deck", "Rank"])
df.to_csv(csv_path, index=False)
print(f"Saved {len(results)} rows to {csv_path}")

# ---- CLEANUP ----
driver.quit()