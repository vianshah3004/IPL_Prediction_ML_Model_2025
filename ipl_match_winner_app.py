import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Define the specific stadiums and teams to use
STADIUMS = [
    'Narendra Modi Stadium, Ahmedabad',
    'M Chinnaswamy Stadium',
    'MA Chidambaram Stadium, Chepauk',
    'Arun Jaitley Stadium',
    'Himachal Pradesh Cricket Association Stadium',
    'Barsapara Cricket Stadium, Guwahati',
    'Rajiv Gandhi International Stadium, Uppal',
    'Sawai Mansingh Stadium',
    'Eden Gardens',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
    'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
    'Wankhede Stadium',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium'
]

TEAMS = [
    'Kolkata Knight Riders',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Royal Challengers Bengaluru',
    'Punjab Kings',
    'Delhi Capitals',
    'Mumbai Indians',
    'Sunrisers Hyderabad',
    'Gujarat Titans',
    'Lucknow Super Giants'
]
# Set page config
st.set_page_config(page_title='IPL Match Winner Predictor', layout="wide")

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff5722;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3f51b5;
        margin-bottom: 1rem;
    }
    .team-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .winner-box {
        background-color: #4CAF50;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-header'>IPL Match Winner Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
This app predicts the winner of an IPL match based on historical data, team strengths, venue statistics, and match conditions.
The model considers head-to-head records, venue advantage, team batting and bowling strengths, recent form, and toss factors.
""")

# Load the model and required data
@st.cache_resource
def load_model_and_data():
    try:
        model_data = pickle.load(open("C:/final year project/ml_miniproject/ipl_match_winner_model.pkl", "rb"))
        
        # Extract model components
        model = model_data['model']
        scaler = model_data['scaler']
        head_to_head_stats = model_data['head_to_head_stats']
        venue_advantage = model_data['venue_advantage']
        team_strength = model_data['team_strength']
        team_form = model_data['team_form']
        match_type_advantage = model_data.get('match_type_advantage', {})  # Handle older model versions
        unique_teams = model_data['unique_teams']
        feature_names = model_data['feature_names']
        
        # Load matches history for venues and match types
        matches_history = pd.read_csv('C:/final year project/ML/matches.csv')
        
        # Get match types
        match_types = ['league', 'qualifier', 'eliminator', 'final']
        if 'match_type' in matches_history.columns:
            match_types = sorted(matches_history['match_type'].dropna().unique())
        
        # Ensure all teams have entries in the team_strength dictionary
        for team in TEAMS:
            if team not in team_strength:
                # Add default values for missing teams
                team_strength[team] = {
                    'batting_strength': 0.5,
                    'bowling_strength': 0.5,
                    'overall_strength': 0.5
                }
            
            # Ensure all teams have entries in team_form
            if team not in team_form:
                team_form[team] = 0.5
                
            # Ensure all teams have entries in match_type_advantage
            if team not in match_type_advantage:
                match_type_advantage[team] = {match_type: 0.5 for match_type in match_types}
        
        # Ensure all venues have entries in venue_advantage
        for venue in STADIUMS:
            if venue not in venue_advantage:
                venue_advantage[venue] = {team: 0.5 for team in TEAMS}
        
        return model, scaler, head_to_head_stats, venue_advantage, team_strength, team_form, match_type_advantage, unique_teams, match_types, feature_names
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        # Return placeholder data for demonstration
        return None, None, {}, {}, {}, {}, {}, [], ['league', 'qualifier', 'eliminator', 'final'], []

# Load model and data
model, scaler, head_to_head_stats, venue_advantage, team_strength, team_form, match_type_advantage, unique_teams, match_types, feature_names = load_model_and_data()

# Function to predict match winner
def predict_match_winner(team1, team2, venue, toss_winner, toss_decision, match_type):
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
    prediction = model.predict_proba(feature_scaled)[0]
    
    # Create feature importance data
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        for i, name in enumerate(feature_names):
            feature_importance[name] = feature[i] * model.feature_importances_[i]
    
    return {
        team1: prediction[1],
        team2: prediction[0]
    }, feature_importance

# Add match type selection to the sidebar
st.sidebar.markdown("<h2 class='sub-header'>Match Information</h2>", unsafe_allow_html=True)

# Team selection - use only the specified teams
team1 = st.sidebar.selectbox('Select Team 1', TEAMS)
team2 = st.sidebar.selectbox('Select Team 2', TEAMS, 
                            index=min(1, len(TEAMS)-1) if len(TEAMS) > 1 else 0)

if team1 == team2:
    st.warning('Please select different teams')
else:
    # Venue selection - use only the specified venues
    venue = st.sidebar.selectbox('Select Venue', STADIUMS)
    
    # Match type selection
    match_type = st.sidebar.selectbox('Match Type', match_types, index=0)
    
    # Toss information
    toss_winner = st.sidebar.radio('Toss Winner', [team1, team2])
    toss_decision = st.sidebar.radio('Toss Decision', ['bat', 'field'])
    
    # Make prediction button
    if st.sidebar.button('Predict Winner'):
        # Make prediction
        prediction, feature_importance = predict_match_winner(team1, team2, venue, toss_winner, toss_decision, match_type)
        
        # Display prediction
        st.markdown("<h2 class='sub-header'>Match Prediction</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Determine winner
        team1_prob = prediction[team1]
        team2_prob = prediction[team2]
        predicted_winner = team1 if team1_prob > team2_prob else team2
        
        with col1:
            st.markdown(f"<div class='team-header'>{team1}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='prediction-box' style='background-color: {'#e8f5e9' if predicted_winner == team1 else '#f5f5f5'};'>"
                f"<h3>Win Probability: {team1_prob:.2%}</h3>"
                f"<p>Batting Strength: {team_strength.get(team1, {}).get('batting_strength', 0.5):.2f}</p>"
                f"<p>Bowling Strength: {team_strength.get(team1, {}).get('bowling_strength', 0.5):.2f}</p>"
                f"<p>Recent Form: {team_form.get(team1, 0.5):.2f}</p>"
                f"<p>Match Type Performance: {match_type_advantage.get(team1, {}).get(match_type, 0.5):.2f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
            
        with col2:
            st.markdown(f"<div class='team-header'>{team2}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='prediction-box' style='background-color: {'#e8f5e9' if predicted_winner == team2 else '#f5f5f5'};'>"
                f"<h3>Win Probability: {team2_prob:.2%}</h3>"
                f"<p>Batting Strength: {team_strength.get(team2, {}).get('batting_strength', 0.5):.2f}</p>"
                f"<p>Bowling Strength: {team_strength.get(team2, {}).get('bowling_strength', 0.5):.2f}</p>"
                f"<p>Recent Form: {team_form.get(team2, 0.5):.2f}</p>"
                f"<p>Match Type Performance: {match_type_advantage.get(team2, {}).get(match_type, 0.5):.2f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Display result
        st.markdown(
            f"<div class='winner-box'>"
            f"Predicted Winner: {predicted_winner}"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Display match details
        st.markdown("<h2 class='sub-header'>Match Details</h2>", unsafe_allow_html=True)
        details = {
            "Teams": f"{team1} vs {team2}",
            "Venue": venue,
            "Match Type": match_type,
            "Toss Winner": toss_winner,
            "Toss Decision": toss_decision,
            "Head-to-Head Advantage": f"{head_to_head_stats.get(team1, {}).get(team2, 0.5):.2f} in favor of {team1}"
        }
        
        for key, value in details.items():
            st.markdown(f"**{key}:** {value}")
        
        # Display factors influencing prediction
        st.markdown("<h2 class='sub-header'>Factors Influencing Prediction</h2>", unsafe_allow_html=True)
        
        if feature_importance:
            # Sort factors by importance
            sorted_factors = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Create dataframe for visualization
            factors_df = pd.DataFrame({
                'Factor': [factor for factor, _ in sorted_factors],
                'Impact': [impact for _, impact in sorted_factors]
            })
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(x='Impact', y='Factor', data=factors_df, ax=ax)
            
            # Color bars based on positive/negative impact
            for i, bar in enumerate(bars.patches):
                if factors_df.iloc[i]['Impact'] < 0:
                    bar.set_facecolor('#ff9999')
                else:
                    bar.set_facecolor('#99ff99')
            
            ax.set_title('Importance of Factors in Prediction')
            st.pyplot(fig)

# Update the explanation section to include match type
with st.expander("How does the prediction work?"):
    st.write("""
    The prediction model considers several factors:
    
    1. **Head-to-Head Record**: Historical performance between the two teams
    2. **Venue Advantage**: How well each team performs at the selected venue
    3. **Team Strength**: Current team strength based on player statistics
       - Batting strength (average, strike rate)
       - Bowling strength (average, economy rate)
    4. **Recent Form**: Performance in recent matches
    5. **Match Type Performance**: How teams perform in different match types (league, qualifier, eliminator, final)
    6. **Toss Advantage**: Impact of winning the toss
    7. **Batting/Bowling First**: Advantage of batting or bowling first
    
    The model was trained on IPL match data from 2008 to 2024, including player statistics and match results.
    """)

# Add information about data sources
with st.expander("Data Sources"):
    st.write("""
    This prediction model uses the following datasets:
    
    1. **Player Statistics**: Yearly batting and bowling statistics for all IPL players
    2. **Match History**: Complete record of all IPL matches from 2008 to 2024
    3. **Auction Data**: Information about player teams and roles for the current season
    
    The model is regularly updated with the latest match results and player performances.
    """)

# Add information about the stadiums and teams
with st.expander("Stadiums and Teams"):
    st.write("### IPL Stadiums")
    for stadium in STADIUMS:
        st.write(f"- {stadium}")
    
    st.write("### IPL Teams")
    for team in TEAMS:
        st.write(f"- {team}")