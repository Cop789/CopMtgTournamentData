import streamlit as st
import pandas as pd
import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title("MTG Tournament Dashboard")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_all_data():
    # Tournament mapping
    tournament_map = pd.read_csv("data/TournamentScraped.csv", dtype=str)
    tournament_map["TournamentNumber"] = tournament_map["TournamentNumber"].astype(str)
    tournament_dict = dict(zip(
        tournament_map["TournamentNumber"],
        tournament_map["TournamentName"]
    ))

    # Load match data
    match_list = []
    match_folder = "data/MatchData"
    match_files = [f for f in os.listdir(match_folder) if f.endswith(".csv")]
    for file in match_files:
        tournament_number = str(re.search(r"\d+", file).group())
        df = pd.read_csv(f"{match_folder}/{file}")
        df["TournamentName"] = tournament_dict[tournament_number]
        match_list.append(df)
    match_df = pd.concat(match_list, ignore_index=True)

    # Load player data
    player_list = []
    player_folder = "data/PlayerData"
    player_files = [f for f in os.listdir(player_folder) if f.endswith(".csv")]
    for file in player_files:
        tournament_number = str(re.search(r"\d+", file).group())
        df = pd.read_csv(f"{player_folder}/{file}")
        df["TournamentName"] = tournament_dict[tournament_number]
        player_list.append(df)
    player_df = pd.concat(player_list, ignore_index=True)

    return match_df, player_df

match_df, player_df = load_all_data()

# -------------------------
# TOURNAMENT SELECTION
# -------------------------

# Keep tournaments in the order they appear
tournament_ordered = match_df["TournamentName"].tolist()
tournament_ordered.reverse()

tournament = st.selectbox(
    "Select Tournament",
    tournament_ordered
)

filtered_matches = match_df[match_df["TournamentName"] == tournament].copy()
filtered_players = player_df[player_df["TournamentName"] == tournament].copy()

# -------------------------
# SPLIT PLAYERS AND DECKS INTO TWO COLUMNS
# -------------------------
PRONOUNS = ["He/Him","She/Her","They/Them", "She/They", "He/They", "Ze/zir", "Other"]
def remove_pronouns(name):
    if pd.isna(name):
        return name
    for p in PRONOUNS:
        name = re.sub(rf"\b{re.escape(p)}\b", "", name, flags=re.IGNORECASE)
    return name.strip()

if 'Players' in filtered_matches.columns:
    filtered_matches[['Player1','Player2']] = (
        filtered_matches['Players']
        .str.split('\n', n=1, expand=True)
    )
    filtered_matches['Player2'] = filtered_matches['Player2'].fillna('Unknown')

    # Remove pronouns, add more pronouns above as needed
    filtered_matches['Player1'] = filtered_matches['Player1'].apply(remove_pronouns)
    filtered_matches['Player2'] = filtered_matches['Player2'].apply(remove_pronouns)

if 'Decks' in filtered_matches.columns:
    filtered_matches[['Deck1','Deck2']] = (
        filtered_matches['Decks']
        .str.split('\n', n=1, expand=True)
    )
    filtered_matches['Deck2'] = filtered_matches['Deck2'].fillna('Unknown')

# -------------------------
# PROCESS WINNER INTO RESULT
# -------------------------
def process_winner(row):
    winner_raw = str(row['Winner'])
    player1 = str(row.get('Player1', 'Unknown')).strip()
    player2 = str(row.get('Player2', 'Unknown')).strip()
    deck1 = str(row.get('Deck1', 'Unknown')).strip()
    deck2 = str(row.get('Deck2', 'Unknown')).strip()

    # Bye
    if 'assigned a bye' in winner_raw.lower() or 'awarded a bye' in winner_raw.lower():
        return pd.Series({'Result':'Bye', 'WinnerPlayer': player1, 'WinnerDeck': deck1})
    
    # Forfeit
    if 'forfeit' in winner_raw.lower():
        if player1.lower() in winner_raw.lower():
            return pd.Series({'Result':'Deck2', 'WinnerPlayer': player2, 'WinnerDeck': deck2})
        else:
            return pd.Series({'Result':'Deck1', 'WinnerPlayer': player1, 'WinnerDeck': deck1})
    
    # Draw
    DrawValue = ["1-1-1 Draw", "1-1-0 Draw", "0-0-3 Draw"]
    if any(draw_val in winner_raw for draw_val in DrawValue):
        return pd.Series({'Result':'Draw', 'WinnerPlayer': None, 'WinnerDeck': None})
    
    # Normal win: "{name} won X-Y-Z"
    win_match = re.match(r'(.+?) won \d+-\d+-\d+', winner_raw)
    if win_match:
        winner_name = win_match.group(1).strip()
        if winner_name.lower() == player1.lower():
            return pd.Series({'Result':'Deck1', 'WinnerPlayer': player1, 'WinnerDeck': deck1})
        elif winner_name.lower() == player2.lower():
            return pd.Series({'Result':'Deck2', 'WinnerPlayer': player2, 'WinnerDeck': deck2})
        else:
            # Fallback
            return pd.Series({'Result':'Deck1', 'WinnerPlayer': winner_name, 'WinnerDeck': deck1})
    
    return pd.Series({'Result':'Unknown', 'WinnerPlayer': None, 'WinnerDeck': None})

filtered_matches[['Result','WinnerPlayer','WinnerDeck']] = filtered_matches.apply(process_winner, axis=1)

# -------------------------
# PLAYER COUNT CARD
# -------------------------
st.metric(label="Number of Players", value=filtered_players.shape[0])

# -------------------------
# MATCHUP TABLE    Can add on later if needed, but not sure if I need to display this data rn
# -------------------------
#st.subheader("Matchup Table")
#display_cols = ['Player1','Player2','Deck1','Deck2','Round','Result','WinnerPlayer','WinnerDeck']
#st.dataframe(filtered_matches[display_cols], hide_index=True)

# -------------------------
# TOURNAMENT STATS
# -------------------------
st.subheader("Tournament Stats")

num_rounds = filtered_matches['Round'].nunique()-3
mirror_matches = filtered_matches[filtered_matches["Deck1"] == filtered_matches["Deck2"]].shape[0]
draws = filtered_matches[filtered_matches["Result"] == "Draw"].shape[0]
byes = filtered_matches[filtered_matches["Result"] == "Bye"].shape[0]
total_matches = filtered_matches.shape[0]

tournament_stats = pd.DataFrame({
    "Metric": ["Number of Rounds", "Mirror Matches", "Draws", "Byes", "Total Matches"],
    "Value": [num_rounds, mirror_matches, draws, byes, total_matches]
})

st.dataframe(tournament_stats, hide_index=True)

# -------------------------
# MOST POPULAR DECKS STATS
# -------------------------
st.subheader("Decks Day 2 and Finishing Stats")

# Get only Round 10 matches
round10_matches = filtered_matches[filtered_matches["Round"] == "Round 10"]

# Stack Deck1 and Deck2 into a single series for counting
all_decks_in_round10 = pd.concat([round10_matches["Deck1"], round10_matches["Deck2"]])

# Count each deck
round10_counts = all_decks_in_round10.value_counts().reset_index()
round10_counts.columns = ["Deck", "Day 2"]

round10_df = pd.DataFrame(round10_counts)

deck_stats = filtered_players.groupby("Deck").agg(
    NumPlayers=("Deck", "count"),
    AverageFinalRank=("Rank", "mean"),
    HighestFinish=("Rank", "min"),
    LowestFinish=("Rank", "max"),
).reset_index()

deck_stats["AverageFinalRank"] = deck_stats["AverageFinalRank"].round(0).astype(int)
deck_stats = deck_stats.merge(round10_df, on="Deck", how="left")
deck_stats["Day 2"] = deck_stats["Day 2"].fillna(0).round(2)
deck_stats["Day 2 %"] = (deck_stats["Day 2"] / deck_stats["NumPlayers"] * 100).round(1)
deck_stats = deck_stats.sort_values(by="NumPlayers", ascending=False)

st.dataframe(deck_stats, hide_index=True)

# -------------------------
# TOP DECKS BY WINRATE
# -------------------------
st.subheader("Decks by Win %")

deck_win_data = []
for deck in filtered_players['Deck'].unique():
    total_matches = filtered_matches[
        (filtered_matches['Deck1'] == deck) | (filtered_matches['Deck2'] == deck)
    ]
    wins = total_matches[total_matches['WinnerDeck'] == deck].shape[0]
    draws = total_matches[total_matches['Result'] == 'Draw'].shape[0]
    losses = total_matches.shape[0] - wins - draws
    win_pct = round((wins / total_matches.shape[0] * 100), 2) if total_matches.shape[0] > 0 else 0
    deck_win_data.append({
        'Deck': deck,
        'Wins': wins,
        'Losses': losses,
        'Draws': draws,
        'Matches': total_matches.shape[0],
        'Win %': win_pct
    })

top_winrate_df = pd.DataFrame(deck_win_data).sort_values(by='Win %', ascending=False)
st.dataframe(top_winrate_df, hide_index=True)

# -------------------------
# WINRATE HEATMAP WITH MULTIPLE METRICS
# -------------------------
st.subheader("Top 10 Decks Matchups(Wins / Matches / Winrate %)")

# Top 10 decks by popularity, change head number to adjust how many decks to show in heatmap
deck_counts = filtered_players['Deck'].value_counts()
top_decks = deck_counts.head(10).index.tolist()

# Filter matches to top decks
heatmap_matches = filtered_matches[
    filtered_matches['Deck1'].isin(top_decks) &
    filtered_matches['Deck2'].isin(top_decks)
].copy()

# Prepare dataframes
winrate = pd.DataFrame(index=top_decks, columns=top_decks, dtype=float)
matches_count = pd.DataFrame(index=top_decks, columns=top_decks, dtype=int)
wins_count = pd.DataFrame(index=top_decks, columns=top_decks, dtype=int)
annot_text = pd.DataFrame(index=top_decks, columns=top_decks, dtype=str)

for d1 in top_decks:
    for d2 in top_decks:
        wins = heatmap_matches[
            ((heatmap_matches['Deck1']==d1) & (heatmap_matches['Deck2']==d2) & (heatmap_matches['Result']=='Deck1')) |
            ((heatmap_matches['Deck1']==d2) & (heatmap_matches['Deck2']==d1) & (heatmap_matches['Result']=='Deck2'))
        ].shape[0]

        matches = heatmap_matches[
            ((heatmap_matches['Deck1']==d1) & (heatmap_matches['Deck2']==d2)) |
            ((heatmap_matches['Deck1']==d2) & (heatmap_matches['Deck2']==d1))
        ].shape[0]

        # Mirror matches: double matches
        if d1 == d2:
            matches *= 2

        wr = (wins / matches * 100) if matches > 0 else 0
        if d1 == d2:
            wr = 50  # Mirror color

        wins_count.loc[d1,d2] = wins
        matches_count.loc[d1,d2] = matches
        annot_text.loc[d1,d2] = f"W:{wins}\nM:{matches}\n{wr:.0f}%"
        winrate.loc[d1,d2] = wr

# Plot heatmap
plt.figure(figsize=(12,10))
sns.heatmap(winrate.loc[top_decks, top_decks],
            annot=annot_text.loc[top_decks, top_decks],
            fmt='', cmap="coolwarm",
            linewidths=0.5, linecolor='gray',
            cbar_kws={'label':'Winrate (%)'}
            )

plt.xlabel("Losing Deck", fontsize=12)
plt.ylabel("Winning Deck", fontsize=12)
plt.title("Deck vs Deck (Wins / Matches / Winrate %)")
st.pyplot(plt)
