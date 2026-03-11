from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import time
import os

# --- Setup ---
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)
TournamentNumber = "408838"  # Replace with actual tournament number
url = f"https://melee.gg/Tournament/View/{TournamentNumber}"
driver.get(url)

save_folder = r"Data\MatchData"
csv_filename = f"matchesData_{TournamentNumber}.csv"

# Ensure folder exists
os.makedirs(save_folder, exist_ok=True)
csv_path = os.path.join(save_folder, csv_filename)

# --- Accept necessary cookies only ---
try:
    necessary_button = wait.until(EC.element_to_be_clickable((By.ID, "necessaryOnlyButton")))
    necessary_button.click()
    print("Accepted necessary cookies")
except:
    print("Cookie popup not found, continuing...")

time.sleep(2)

# --- Scrape standings table ---
standings_data = []
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tournament-pairings-table tbody tr")))

while True:
    rows = driver.find_elements(By.CSS_SELECTOR, "#tournament-pairings-table tbody tr")
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) >= 2:
            standings_data.append({
                "Players": cols[1].text.strip(),
                "Decks": cols[2].text.strip()
            })

    try:
        next_button = driver.find_element(By.ID, "tournament-pairings-table_next")
        if "disabled" in next_button.get_attribute("class"):
            break
        else:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
            wait.until(EC.element_to_be_clickable(next_button))
            ActionChains(driver).move_to_element(next_button).click().perform()
            time.sleep(1)
    except:
        break

standings_df = pd.DataFrame(standings_data)
standings_df.to_csv("standings.csv", index=False)
print("Standings table saved as standings.csv")

# --- Scrape matches table per round with pagination ---
matches_data = []
round_buttons = driver.find_elements(By.CSS_SELECTOR, "#pairings-round-selector-container button.round-selector")

for round_button in round_buttons:
    round_name = round_button.get_attribute("data-name")

    # Click the round if not already active
    if "active" not in round_button.get_attribute("class"):
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", round_button)
        wait.until(EC.element_to_be_clickable(round_button))
        ActionChains(driver).move_to_element(round_button).click().perform()
        time.sleep(1)  # wait for table to update

    print(f"Scraping matches for {round_name}...")

    # --- Reset to page 1 for this round ---
    while True:
        try:
            first_page = driver.find_element(By.CSS_SELECTOR, "#tournament-pairings-table_paginate a[data-dt-idx='1']")
            if "current" in first_page.get_attribute("class"):
                break
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", first_page)
            wait.until(EC.element_to_be_clickable(first_page))
            ActionChains(driver).move_to_element(first_page).click().perform()
            time.sleep(1)
        except:
            break

    # --- Loop through pagination ---
    while True:
        # Wait for table rows
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#tournament-pairings-table tbody tr")))
        match_rows = driver.find_elements(By.CSS_SELECTOR, "#tournament-pairings-table tbody tr")

        for row in match_rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 3:
                matches_data.append({
                    "Round": round_name,
                    "Table": cols[0].text.strip(),
                    "Players": cols[1].text.strip(),
                    "Decks": cols[2].text.strip(),
                    "Winner": cols[3].text.strip()
                })

        # --- Check if we can go to next page ---
        try:
            next_button = driver.find_element(By.ID, "tournament-pairings-table_next")
            # Always re-check class after the table refresh
            if "disabled" in next_button.get_attribute("class"):
                break
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
            wait.until(EC.element_to_be_clickable(next_button))
            ActionChains(driver).move_to_element(next_button).click().perform()
            time.sleep(1)  # wait for table update
        except:
            break

df = pd.DataFrame(matches_data, columns=["Round", "Table", "Players", "Decks", "Winner"])
df.to_csv(csv_path, index=False)
print(f"Saved {len(matches_data)} rows to {csv_path}")

driver.quit()