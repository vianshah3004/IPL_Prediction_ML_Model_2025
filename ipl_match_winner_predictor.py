import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load datasets
print("Loading datasets...")
player_stats = pd.read_csv('C:/final year project/ML/cricket_data_2025.csv')
matches_history = pd.read_csv('C:/final year project/ML/matches.csv')
auction_players = pd.read_csv('C:/final year project/ML/ipl_2025_auction_players.csv')

print(f"Player Stats Shape: {player_stats.shape}")
print(f"Matches History Shape: {matches_history.shape}")
print(f"Auction Players Shape: {auction_players.shape}")

# Display first few rows of each dataset
print("\nPlayer Stats Preview:")
print(player_stats.head())

print("\nMatches History Preview:")
print(matches_history.head())

print("\nAuction Players Preview:")
print(auction_players.head())

# Data Preprocessing
print("\nPreprocessing data...")

# Convert date to datetime
matches_history['date'] = pd.to_datetime(matches_history['date'], format='%d-%m-%Y', errors='coerce')
# If the above fails, try alternative format
if matches_history['date'].isna().all():
    matches_history['date'] = pd.to_datetime(matches_history['date'], errors='coerce')

# Extract year from date
matches_history['year'] = matches_history['date'].dt.year

# Standardize team names (some teams might have changed names over the years)
team_name_mapping = {
    'Delhi Capitals': 'Delhi Daredevils',
    'Delhi Daredevils': 'Delhi Capitals',
    'Punjab Kings': 'Kings XI Punjab',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Rising Pune Supergiants': 'Rising Pune Supergiant'
}

# Function to standardize team names
def standardize_team_name(team_name):
    # Return the standardized name if it exists, otherwise return the original name
    return team_name_mapping.get(team_name, team_name)

# Apply standardization to team columns
for col in ['team1', 'team2', 'toss_winner', 'winner']:
    if col in matches_history.columns:
        matches_history[col] = matches_history[col].apply(lambda x: standardize_team_name(x) if pd.notna(x) else x)

# Get unique teams after standardization
unique_teams = sorted(list(set(
    list(matches_history['team1'].dropna().unique()) + 
    list(matches_history['team2'].dropna().unique())
)))

print(f"Unique teams in the dataset: {unique_teams}")

# After loading the datasets, add this code to verify all 10 teams are present
print("\nVerifying all 10 IPL teams are present...")
expected_teams = [
    'Mumbai Indians', 
    'Chennai Super Kings', 
    'Royal Challengers Bangalore', 
    'Kolkata Knight Riders', 
    'Delhi Capitals', 
    'Sunrisers Hyderabad', 
    'Punjab Kings', 
    'Rajasthan Royals',
    'Gujarat Titans',
    'Lucknow Super Giants'
]

# Check if all expected teams are in the dataset
missing_teams = [team for team in expected_teams if team not in unique_teams]
if missing_teams:
    print(f"Warning: The following teams are missing from the dataset: {missing_teams}")
    # Add missing teams to unique_teams
    for team in missing_teams:
        if team not in unique_teams:
            unique_teams.append(team)
    unique_teams = sorted(unique_teams)
    print(f"Added missing teams. Total teams: {len(unique_teams)}")
else:
    print(f"All 10 IPL teams are present. Total teams: {len(unique_teams)}")

# Feature Engineering
print("\nEngineering features...")

# 1. Head-to-head statistics
def create_head_to_head_stats(df):
    head_to_head = {}
    
    for team1 in unique_teams:
        head_to_head[team1] = {}
        for team2 in unique_teams:
            if team1 != team2:
                # Matches where team1 is team1 and team2 is team2
                matches1 = df[(df['team1'] == team1) & (df['team2'] == team2)]
                team1_wins1 = matches1[matches1['winner'] == team1].shape[0]
                
                # Matches where team1 is team2 and team2 is team1
                matches2 = df[(df['team1'] == team2) & (df['team2'] == team1)]
                team1_wins2 = matches2[matches2['winner'] == team1].shape[0]
                
                total_matches = matches1.shape[0] + matches2.shape[0]
                if total_matches > 0:
                    win_rate = (team1_wins1 + team1_wins2) / total_matches
                else:
                    win_rate = 0.5  # Default if no matches
                
                head_to_head[team1][team2] = win_rate
    
    return head_to_head

head_to_head_stats = create_head_to_head_stats(matches_history)

# 2. Venue advantage
def calculate_venue_advantage(df):
    venue_advantage = {}
    venues = df['venue'].dropna().unique()
    
    for venue in venues:
        venue_matches = df[df['venue'] == venue]
        venue_advantage[venue] = {}
        
        for team in unique_teams:
            # Matches where team played at this venue
            team_matches = venue_matches[(venue_matches['team1'] == team) | (venue_matches['team2'] == team)]
            team_wins = team_matches[team_matches['winner'] == team].shape[0]
            
            if team_matches.shape[0] > 0:
                win_rate = team_wins / team_matches.shape[0]
            else:
                win_rate = 0.5  # Default if no matches
                
            venue_advantage[venue][team] = win_rate
    
    return venue_advantage

venue_advantage = calculate_venue_advantage(matches_history)

# 3. Team strength calculation based on player stats
def calculate_team_strength(player_stats_df, auction_df, year=2024):
    # Ensure column names are correctly formatted
    player_stats_df.columns = [col.strip().lower().replace(' ', '_') for col in player_stats_df.columns]
    auction_df.columns = [col.strip().lower().replace(' ', '_') for col in auction_df.columns]
    
    # Convert numeric columns to proper numeric types
    numeric_columns = ['batting_average', 'batting_strike_rate', 'bowling_average', 'economy_rate']
    for col in numeric_columns:
        if col in player_stats_df.columns:
            # Replace 'No stats' with NaN
            player_stats_df[col] = player_stats_df[col].replace('No stats', np.nan)
            
            # Convert to numeric, forcing errors to NaN
            player_stats_df[col] = pd.to_numeric(player_stats_df[col], errors='coerce')
    
    # Filter player stats for the given year or the most recent year
    if year in player_stats_df['year'].unique():
        year_stats = player_stats_df[player_stats_df['year'] == year]
    else:
        max_year = player_stats_df['year'].max()
        year_stats = player_stats_df[player_stats_df['year'] == max_year]
    
    # Get current team for each player
    player_teams = {}
    for _, row in auction_df.iterrows():
        player_name = row['players'].strip() if 'players' in auction_df.columns else None
        team = row['team'].strip() if 'team' in auction_df.columns else None
        if player_name and team:
            player_teams[player_name] = team
    
    # Calculate team batting and bowling strength
    team_strength = {}
    
    for team in auction_df['team'].unique():
        team_players = [player for player, t in player_teams.items() if t == team]
        
        # Get stats for team players
        team_player_stats = year_stats[year_stats['player_name'].str.strip().isin([p.strip() for p in team_players])]
        
        # Batting strength metrics (handle potential missing columns)
        batting_avg = team_player_stats['batting_average'].mean() if 'batting_average' in team_player_stats.columns else 0
        batting_sr = team_player_stats['batting_strike_rate'].mean() if 'batting_strike_rate' in team_player_stats.columns else 0
        
        # Bowling strength metrics (handle potential missing columns)
        bowling_avg = team_player_stats['bowling_average'].mean() if 'bowling_average' in team_player_stats.columns else 100
        economy = team_player_stats['economy_rate'].mean() if 'economy_rate' in team_player_stats.columns else 10
        
        # Handle NaN values
        batting_avg = 0 if np.isnan(batting_avg) else batting_avg
        batting_sr = 0 if np.isnan(batting_sr) else batting_sr
        bowling_avg = 100 if np.isnan(bowling_avg) else bowling_avg  # Higher is worse for bowling avg
        economy = 10 if np.isnan(economy) else economy  # Higher is worse for economy
        
        # Normalize and combine metrics
        batting_strength = (batting_avg / 50) * 0.7 + (batting_sr / 150) * 0.3
        bowling_strength = (1 - (bowling_avg / 40)) * 0.7 + (1 - (economy / 12)) * 0.3
        
        # Ensure values are within reasonable range
        batting_strength = max(0, min(1, batting_strength))
        bowling_strength = max(0, min(1, bowling_strength))
        
        team_strength[team] = {
            'batting_strength': batting_strength,
            'bowling_strength': bowling_strength,
            'overall_strength': (batting_strength + bowling_strength) / 2
        }
    
    return team_strength

team_strength = calculate_team_strength(player_stats, auction_players)

# 4. Toss advantage and batting first advantage
def calculate_toss_and_batting_advantage(df):
    # Toss advantage
    toss_matches = df.dropna(subset=['toss_winner', 'winner'])
    toss_wins = toss_matches[toss_matches['toss_winner'] == toss_matches['winner']].shape[0]
    toss_advantage = toss_wins / toss_matches.shape[0] if toss_matches.shape[0] > 0 else 0.5
    
    # Batting first advantage
    batting_first_wins = 0
    total_valid_matches = 0
    
    for _, match in df.dropna(subset=['toss_winner', 'toss_decision', 'winner']).iterrows():
        total_valid_matches += 1
        
        # Determine who batted first
        if match['toss_decision'].lower() in ['bat', 'batting']:
            batting_first_team = match['toss_winner']
        else:
            batting_first_team = match['team1'] if match['toss_winner'] == match['team2'] else match['team2']
        
        # Check if batting first team won
        if batting_first_team == match['winner']:
            batting_first_wins += 1
    
    batting_first_advantage = batting_first_wins / total_valid_matches if total_valid_matches > 0 else 0.5
    
    return toss_advantage, batting_first_advantage

toss_advantage, batting_first_advantage = calculate_toss_and_batting_advantage(matches_history)

print(f"Toss Advantage: {toss_advantage:.4f}")
print(f"Batting First Advantage: {batting_first_advantage:.4f}")

# 5. Recent form (last 5 matches)
def calculate_recent_form(df):
    team_form = {}
    
    for team in unique_teams:
        # Get matches where the team played
        team_matches = df[(df['team1'] == team) | (df['team2'] == team)].sort_values('date')
        
        # Calculate win/loss in last 5 matches
        recent_matches = team_matches.tail(5)
        wins = recent_matches[recent_matches['winner'] == team].shape[0]
        
        if recent_matches.shape[0] > 0:
            form = wins / recent_matches.shape[0]
        else:
            form = 0.5  # Default if no recent matches
        
        team_form[team] = form
    
    return team_form

team_form = calculate_recent_form(matches_history)

# Add match type advantage calculation after the recent form calculation

# 6. Match type advantage (how teams perform in different match types: league, qualifier, eliminator, final)
def calculate_match_type_advantage(df):
    match_type_advantage = {}
    
    # Check if match_type column exists
    if 'match_type' not in df.columns:
        print("Warning: match_type column not found in dataset")
        return match_type_advantage
    
    # Get unique match types
    match_types = df['match_type'].dropna().unique()
    
    for team in unique_teams:
        match_type_advantage[team] = {}
        
        for match_type in match_types:
            # Get matches of this type where the team played
            type_matches = df[(df['match_type'] == match_type) & ((df['team1'] == team) | (df['team2'] == team))]
            
            # Calculate win rate in this match type
            if type_matches.shape[0] > 0:
                wins = type_matches[type_matches['winner'] == team].shape[0]
                win_rate = wins / type_matches.shape[0]
            else:
                win_rate = 0.5  # Default if no matches of this type
            
            match_type_advantage[team][match_type] = win_rate
    
    return match_type_advantage

match_type_advantage = calculate_match_type_advantage(matches_history)
print("\nCalculated match type advantage for teams")


# Prepare dataset for model training
print("\nPreparing dataset for model training...")

# Create features for each match
def prepare_match_features(df, head_to_head, venue_adv, team_str, team_form, match_type_adv=None):
    features = []
    labels = []
    match_ids = []
    
    for _, match in df.dropna(subset=['team1', 'team2', 'winner']).iterrows():
        team1 = match['team1']
        team2 = match['team2']
        venue = match['venue'] if pd.notna(match['venue']) else "Unknown Venue"
        toss_winner = match['toss_winner'] if pd.notna(match['toss_winner']) else team1
        toss_decision = match['toss_decision'] if pd.notna(match['toss_decision']) else 'bat'
        match_type = match['match_type'] if pd.notna(match.get('match_type')) else 'league'
        
        # Skip if teams are not in our standardized list
        if team1 not in unique_teams or team2 not in unique_teams:
            continue
        
        # Head-to-head advantage
        h2h_advantage = head_to_head.get(team1, {}).get(team2, 0.5)
        
        # Venue advantage
        venue_adv_team1 = venue_adv.get(venue, {}).get(team1, 0.5)
        venue_adv_team2 = venue_adv.get(venue, {}).get(team2, 0.5)
        
        # Team strength
        team1_batting = team_str.get(team1, {}).get('batting_strength', 0.5)
        team1_bowling = team_str.get(team1, {}).get('bowling_strength', 0.5)
        team2_batting = team_str.get(team2, {}).get('batting_strength', 0.5)
        team2_bowling = team_str.get(team2, {}).get('bowling_strength', 0.5)
        
        # Recent form
        team1_form = team_form.get(team1, 0.5)
        team2_form = team_form.get(team2, 0.5)
        
        # Match type advantage
        team1_match_type_adv = match_type_adv.get(team1, {}).get(match_type, 0.5) if match_type_adv else 0.5
        team2_match_type_adv = match_type_adv.get(team2, {}).get(match_type, 0.5) if match_type_adv else 0.5
        
        # Toss advantage
        toss_adv_team1 = 1 if toss_winner == team1 else 0
        
        # Batting first
        if toss_winner == team1:
            batting_first_team1 = 1 if str(toss_decision).lower() in ['bat', 'batting'] else 0
        else:
            batting_first_team1 = 0 if str(toss_decision).lower() in ['bat', 'batting'] else 1
        
        # Create feature vector
        feature = [
            h2h_advantage,
            venue_adv_team1, venue_adv_team2,
            team1_batting, team1_bowling,
            team2_batting, team2_bowling,
            team1_form, team2_form,
            team1_match_type_adv, team2_match_type_adv,
            toss_adv_team1,
            batting_first_team1
        ]
        
        features.append(feature)
        
        # Label: 1 if team1 wins, 0 if team2 wins
        label = 1 if match['winner'] == team1 else 0
        labels.append(label)
        
        # Store match ID for reference
        match_ids.append(match['id'] if 'id' in match else 0)
    
    return np.array(features), np.array(labels), match_ids

# Update the call to prepare_match_features to include match_type_advantage
X, y, match_ids = prepare_match_features(matches_history, head_to_head_stats, venue_advantage, team_strength, team_form, match_type_advantage)

print(f"Total matches with complete data: {len(X)}")

# Define feature names for all models
feature_names = [
    'Head-to-Head Advantage',
    'Venue Advantage Team1', 'Venue Advantage Team2',
    'Team1 Batting Strength', 'Team1 Bowling Strength',
    'Team2 Batting Strength', 'Team2 Bowling Strength',
    'Team1 Recent Form', 'Team2 Recent Form',
    'Team1 Match Type Advantage', 'Team2 Match Type Advantage',
    'Toss Advantage Team1',
    'Batting First Team1'
]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Train models
print("\nTraining models...")

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Evaluate the best model
print("\nEvaluating the best model...")
if rf_accuracy > lr_accuracy:
    best_model = rf_model
    best_pred = rf_pred
    print("Random Forest is the best model")
else:
    best_model = lr_model
    best_pred = lr_pred
    print("Logistic Regression is the best model")

print("\nClassification Report:")
print(classification_report(y_test, best_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance (for Random Forest)
if isinstance(best_model, RandomForestClassifier):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Save the model and necessary data for predictions
print("\nSaving the model and data...")
# Update the model_data dictionary to include match_type_advantage
model_data = {
    'model': best_model,
    'scaler': scaler,
    'head_to_head_stats': head_to_head_stats,
    'venue_advantage': venue_advantage,
    'team_strength': team_strength,
    'team_form': team_form,
    'match_type_advantage': match_type_advantage,
    'unique_teams': unique_teams,
    'feature_names': feature_names
}

filename = "ipl_match_winner_model.pkl"
pickle.dump(model_data, open(filename, "wb"))
print(f"Model and data saved as {filename}")

# Function to predict match winner
# Update the predict_match_winner function to include match type
def predict_match_winner(team1, team2, venue, toss_winner, toss_decision, match_type='league'):
    # Get head-to-head advantage
    h2h_advantage = head_to_head_stats.get(team1, {}).get(team2, 0.5)
    
    # Get venue advantage
    venue_adv_team1 = venue_advantage.get(venue, {}).get(team1, 0.5)
    venue_adv_team2 = venue_advantage.get(venue, {}).get(team2, 0.5)
    
    # Get team strength
    team1_batting = team_strength.get(team1, {}).get('batting_strength', 0.5)
    team1_bowling = team_strength.get(team1, {}).get('bowling_strength', 0.5)
    team2_batting = team_strength.get(team2, {}).get('batting_strength', 0.5)
    team2_bowling = team_strength.get(team2, {}).get('bowling_strength', 0.5)
    
    # Get recent form
    team1_form = team_form.get(team1, 0.5)
    team2_form = team_form.get(team2, 0.5)
    
    # Get match type advantage
    team1_match_type_adv = match_type_advantage.get(team1, {}).get(match_type, 0.5)
    team2_match_type_adv = match_type_advantage.get(team2, {}).get(match_type, 0.5)
    
    # Toss advantage
    toss_adv_team1 = 1 if toss_winner == team1 else 0
    
    # Batting first
    if toss_winner == team1:
        batting_first_team1 = 1 if toss_decision.lower() in ['bat', 'batting'] else 0
    else:
        batting_first_team1 = 0 if toss_decision.lower() in ['bat', 'batting'] else 1
    
    # Create feature vector
    feature = [
        h2h_advantage,
        venue_adv_team1, venue_adv_team2,
        team1_batting, team1_bowling,
        team2_batting, team2_bowling,
        team1_form, team2_form,
        team1_match_type_adv, team2_match_type_adv,
        toss_adv_team1,
        batting_first_team1
    ]
    
    # Scale the feature
    feature_scaled = scaler.transform([feature])
    
    # Make prediction
    prediction = best_model.predict_proba(feature_scaled)[0]
    
    return {
        team1: prediction[1],
        team2: prediction[0]
    }

# Example prediction
# Update the example prediction to include match type
print("\nExample prediction:")
example_teams = [team for team in unique_teams if team in team_strength][:2]
if len(example_teams) >= 2:
    team1, team2 = example_teams[0], example_teams[1]
    example_venue = list(venue_advantage.keys())[0] if venue_advantage else "Unknown Venue"
    
    prediction = predict_match_winner(team1, team2, example_venue, team1, 'bat', 'league')
    print(f"Match: {team1} vs {team2} at {example_venue} (League Match)")
    print(f"Toss: {team1} won and chose to bat")
    print(f"Win probability for {team1}: {prediction[team1]:.2%}")
    print(f"Win probability for {team2}: {prediction[team2]:.2%}")
    print(f"Predicted winner: {team1 if prediction[team1] > prediction[team2] else team2}")

else:
    print("Not enough teams to make a prediction.")