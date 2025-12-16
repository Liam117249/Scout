import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="FIFA 22 Smart Scout", page_icon="⚽", layout="wide")

# DICTIONARY: Mapped based on your actual data analysis
# Cluster 2 was the best (Technicians), Cluster 1 was Defenders, etc.
CLUSTER_NAMES = {
    0: "⚙️ The Engine (Balanced Midfielder/Fullback)",
    1: "🛡️ The Stopper (Center Back)",
    2: "⭐ The Elite Technician (Playmaker/Star)",
    3: "⚡ The Speedster (Attacker/Winger)",
    4: "🌱 The Prospect (Developing/Lower Tier)"
}

# --- 2. LOAD DATA & MODELS ---
@st.cache_data
def load_data():
    try:
        # Load the CSV you uploaded
        df = pd.read_csv("fifa_cleaned.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'fifa_cleaned.csv' not found. Please put it in the same folder.")
        return pd.DataFrame()

@st.cache_resource
def load_models():
    try:
        with open('kmeans_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model files (.pkl) not found. Please upload them.")
        return None, None

df = load_data()
model, scaler = load_models()

# --- 3. SIDEBAR & NAVIGATION ---
st.sidebar.title("⚽ FIFA Scout AI")
st.sidebar.info("Navigate the tool below:")
menu = st.sidebar.radio("Go to:", ["🏠 Project Overview", "🔍 Find Cheaper Alternative", "🤖 Predict Player Style"])

# --- 4. MAIN PAGES ---

if menu == "🏠 Project Overview":
    st.title("FIFA 22: Unsupervised Learning Scout")
    st.markdown("""
    ### 🎯 The Problem
    Top players (like **Messi** or **Mbappé**) are too expensive for most clubs.
    
    ### 💡 The Solution
    We used **K-Means Clustering** to group players by **Playing Style** rather than just position.
    This allows scouts to find affordable players who have the exact same statistical profile as the stars.
    """)

    if not df.empty:
        st.subheader("📊 The 5 Player Archetypes Found")
        st.write("Our model identified these distinct groups in the dataset:")
        
        # Calculate stats to show the professor you understand the data
        features = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
        summary = df.groupby('cluster')[features].mean().reset_index()
        summary['Cluster Name'] = summary['cluster'].map(CLUSTER_NAMES)
        
        # Reorder columns for readability
        st.dataframe(summary[['Cluster Name', 'pace', 'shooting', 'passing', 'defending']].style.background_gradient(cmap="Greens"))

elif menu == "🔍 Find Cheaper Alternative":
    st.header("🔍 Moneyball: Find a Cheaper Alternative")
    st.write("Select a star player to find a 'hidden gem' with the same playing style.")

    if not df.empty:
        # Get list of players sorted by value so famous ones are easy to find
        sorted_players = df.sort_values(by='value_eur', ascending=False)['short_name'].head(500).tolist()
        star_player = st.selectbox("Select a Star Player:", sorted_players)

        if st.button("Find Alternatives"):
            # 1. Get the Star's Data
            star_row = df[df['short_name'] == star_player].iloc[0]
            star_cluster = star_row['cluster']
            star_value = star_row['value_eur']
            star_wage = star_row['wage_eur']

            # 2. Display Star Info
            st.info(f"**{star_player}** belongs to **{CLUSTER_NAMES[star_cluster]}**.")
            st.write(f"💰 Market Value: €{star_value:,.0f} | Wage: €{star_wage:,.0f}")

            # 3. Find Matches (Same Cluster, Lower Price)
            # Logic: Same Cluster AND Value is between 1% and 50% of the star's price
            matches = df[
                (df['cluster'] == star_cluster) & 
                (df['value_eur'] < star_value * 0.5) &  # At least 50% cheaper
                (df['value_eur'] > 0) # Remove glitchy 0 value players
            ].sort_values(by='overall', ascending=False).head(5)

            st.subheader(f"✅ Recommended Alternatives (Save 50%+)")
            for i, row in matches.iterrows():
                savings = star_value - row['value_eur']
                st.success(f"**{row['short_name']}** (Overall: {row['overall']}) - Value: €{row['value_eur']:,.0f} (Save €{savings:,.0f}!)")

elif menu == "🤖 Predict Player Style":
    st.header("🤖 AI Scout: Predict New Player Style")
    st.write("Enter raw statistics to see which Archetype this player fits into.")

    # Create 2 columns for sliders
    col1, col2 = st.columns(2)
    
    with col1:
        p_pace = st.slider("Pace", 0, 100, 70)
        p_shoot = st.slider("Shooting", 0, 100, 60)
        p_pass = st.slider("Passing", 0, 100, 65)
    
    with col2:
        p_drib = st.slider("Dribbling", 0, 100, 70)
        p_def = st.slider("Defending", 0, 100, 50)
        p_phys = st.slider("Physical", 0, 100, 60)

    if st.button("Predict Archetype"):
        if model is not None and scaler is not None:
            # 1. Prepare Input (must match the order you trained on!)
            # Note: Ensure these columns match exactly what you used in 'fit' in your notebook
            input_data = pd.DataFrame([[p_pace, p_shoot, p_pass, p_drib, p_def, p_phys]], 
                                      columns=['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic'])
            
            # 2. Scale
            scaled_input = scaler.transform(input_data)
            
            # 3. Predict
            cluster_id = model.predict(scaled_input)[0]
            
            # 4. Result
            st.balloons()
            st.markdown(f"### 🎯 Result: {CLUSTER_NAMES[cluster_id]}")
            st.info("This player fits the profile calculated by our K-Means algorithm.")
        else:
            st.error("Model not loaded. Cannot predict.")
