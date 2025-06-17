# 🏏 IPL Match Winner Predictor

**IPL Match Winner Predictor** is a machine learning-based web application built using Streamlit that predicts the winner of an IPL match based on real-time inputs and historical performance data. It leverages advanced data preprocessing, feature engineering, and a trained classification model to deliver accurate predictions for upcoming IPL matches.

---

## 👨‍💻 Built By

- **Vian Shah**  
- **Shloka Shetiya**  
- **Tirrth Mistry**

---

## ✨ Key Features

- Predicts match winner between any two IPL teams.
- Uses advanced historical features such as:
  - Head-to-head records
  - Venue performance
  - Team strengths (batting & bowling)
  - Toss impact
  - Match type (league, qualifier, eliminator, final)
  - Recent form of both teams
- Provides win probabilities for each team.
- Highlights key factors influencing the prediction via a visual bar chart.

---

## 🎯 Model Insights

- **Algorithms Used**: Random Forest Classifier & Logistic Regression
- **Best Model Chosen**: Random Forest (based on higher accuracy)
- **Error Metrics**:
  - MAE: *~0.3497*
  - MSE: *~0.1682*
  - RMSE: *~0.4102*

---

## 🧪 Sample Output

![Image](https://github.com/user-attachments/assets/83120e25-fb64-4ded-8e3f-66c92a41db5c)

---

### 🔹 Top Influencing Factors
- Team Batting Strength  
- Venue Advantage  
- Toss Decision  
- Recent Form

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (pandas, NumPy)
- **Modeling**: Scikit-learn (Random Forest Classifier)
- **Visualization**: matplotlib, seaborn
- **Data**: IPL Match Data (2008–2024), 2025 Player Stats & Auction Data

---

## 🚀 Getting Started

### 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/vianshah3004/Imaginate.git
   cd Imaginate
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the following CSV files are present in the project directory:
    - matches.csv
    - cricket_data_2025.csv
    - ipl_2025_auction_players.csv

5. Run the model generation script to create the trained model file (ipl_match_winner_model.pkl):
   ```bash
   python ipl_match_winner_predictor.py
   ```
   This script will:
   - Load and preprocess IPL datasets
   - Engineer match features
   - Train and evaluate ML models
   - Save the best model and metadata as ipl_match_winner_model.pkl

6. Launch the Streamlit app for predictions:
   ```bash
   streamlit run ipl_match_winner_app.py
   ```

---

## 🙌 Acknowledgements
Grateful for the collaboration and teamwork with Shloka Shetiya and Tirrth Mistry — this project wouldn't have been possible without them! 


