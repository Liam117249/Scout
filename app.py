import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. SETUP AND CONFIGURATION ---
st.set_page_config(
    page_title="FIFA 22 Smart Scout",
    page_icon="⚽",
    layout="centered"
)

# Start Session State (Just like your Midterm)
if 'page' not in st.session_state:
    st.session_state['page'] = 'welcome'

# --- 2. LOAD FILES ---
@st.cache_data
def load_data():
    # Load the cleaned data
    df = pd.read_csv("fifa_cleaned.csv")
    return df

@st.cache_resource
def load_tools():
    # Load the model and scaler
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    df = load_data()
    model, scaler = load_tools()
except FileNotFoundError:
    st.error("❌ Error: Files not found! Make sure 'fifa_cleaned.csv', 'kmeans_model.pkl', and 'scaler.pkl' are in the folder.")
    st.stop()

# --- 3. CUSTOM CSS (Referenced from your Midterm, changed to Blue Theme) ---
st.markdown("""
    <style>
    /* Main Background - Football Blue Gradient */
    .stApp { 
        background: linear-gradient(to bottom right, #e0f2fe, #1e40af); 
        background-attachment: fixed; 
    }

    /* Content Box */
    .block-container { 
        background-color: rgba(255, 255, 255, 0.95); 
        border-radius: 20px; 
        padding: 2rem; 
        margin-top: 2rem; 
        border: 2px solid #60a5fa; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
    }

    /* Header Box */
    .header-box { 
        background-color: #1e3a8a; 
        padding: 20px; 
        border-radius: 15px; 
        margin-bottom: 30px; 
        text-align: center; 
        border: 2px solid #fbbf24; /* Gold border for FIFA feel */
    }
    .header-box h1 { 
        color: white !important; 
        font-family: 'Sans-Serif'; 
        font-size: 32px; 
        margin-bottom: 10px; 
    }
    .header-box h3 { 
        color: #bfdbfe !important; 
        font-size: 18px; 
        font-weight: normal; 
    }

    /* Welcome Screen */
    .welcome-box { 
        text-align: center; 
        padding: 40px; 
    }
    .welcome-icon { 
        font-size: 80px; 
        margin-bottom: 20px; 
        display: block; 
    }

    /* Buttons */
    .stButton>button { 
        background-color: #1e3a8a; 
        color: white; 
        font-size: 20px; 
        border-radius: 12px; 
        height: 55px; 
        border: 2px solid #fbbf24; 
        width: 100%; 
    }
    .stButton>button:hover { 
        background-color: #2563eb; 
        border-color: white; 
        transform: scale(1.02); 
    }

    /* Sidebar */
    [data-testid="stSidebar"] { 
        background-color: #1e3a8a; 
        border-right: 2px solid #fbbf24; 
    }
    [data-testid="stSidebar"] * { 
        color: white !important; 
    }

    /* Tip Box / Result Box */
    .result-box { 
        background-color: #eff6ff; 
        border-left: 5px solid #1e3a8a; 
        padding: 15px; 
        border-radius: 5px; 
        margin-top: 15px; 
    }
    </style>
""", unsafe_allow_html=True)


# --- 4. FUNCTIONS TO SHOW PAGES ---

def show_welcome():
    # Welcome Screen Logic
    st.markdown("""
        <div class="welcome-box">
            <span class="welcome-icon">⚽</span>
            <div class="header-box">
                <h1>Welcome to FIFA Smart Scout</h1>
            </div>
            <h3>AI-Powered Football Player Clustering & Scouting</h3>
            <br>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Scouting"):
            st.session_state['page'] = 'main'
            st.rerun()

def show_main_app():
    # Sidebar Navigation
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/FIFA_22_Logo.png/640px-FIFA_22_Logo.png", width=150)
    st.sidebar.header("Navigation")
    
    menu = st.sidebar.radio(
        "", 
        ["🏠 Project Overview", "📊 Cluster Analysis", "🔍 Player Scout Tool", "ℹ️ About Me"]
    )

    # === SECTION 1: PROJECT OVERVIEW ===
    if menu == "🏠 Project Overview":
        st.markdown("""
            <div class="header-box">
                <h1>FIFA 22 Player Clustering</h1>
                <h3>Unsupervised Learning Final Project</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.image("https://thumbs.dreamstime.com/b/football-soccer-players-silhouette-vector-illustration-sport-concept-white-background-131756534.jpg", use_container_width=True)
        
        st.markdown("""
        <div class="result-box">
            <b>🎯 Project Goal:</b><br>
            To group football players into distinct <b>playing styles</b> using AI (K-Means Clustering), 
            instead of relying on generic position names like 'ST' or 'CB'.
        </div>
        """, unsafe_allow_html=True)

    # === SECTION 2: CLUSTER ANALYSIS ===
    elif menu == "📊 Cluster Analysis":
        st.title("📊 Visualize Player Groups")
        
        # User Controls
        col1, col2 = st.columns(2)
        with col1:
            # Filter columns to only show numeric ones for the graph
            numeric_cols = [c for c in df.columns if c not in ['short_name', 'cluster']]
            x_axis = st.selectbox("X-Axis (Stat):", numeric_cols, index=2) # Default Pace
        with col2:
            y_axis = st.selectbox("Y-Axis (Stat):", numeric_cols, index=3) # Default Shooting

        # Graph
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='cluster', palette='viridis', s=60, alpha=0.7, ax=ax)
        plt.title(f"{x_axis.title()} vs {y_axis.title()}")
        plt.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

        # Interpretation Table
        st.subheader("📝 Cluster Profiles (Average Stats)")
        key_stats = ['overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
        cluster_stats = df.groupby('cluster')[key_stats].mean().reset_index()
        st.dataframe(cluster_stats.style.background_gradient(cmap="Blues"))

    # === SECTION 3: SCOUT TOOL ===
    elif menu == "🔍 Player Scout Tool":
        st.title("🔍 Scout a Player")
        st.markdown("Find a player's style and discover similar alternatives.")
        
        player_name = st.selectbox("Type a player's name:", df['short_name'].unique())

        if st.button("Analyze Player"):
            # Logic to find player
            player_data = df[df['short_name'] == player_name].iloc[0]
            cluster_id = player_data['cluster']
            
            # Show Result
            st.success(f"✅ **{player_name}** belongs to **Cluster {cluster_id}**")
            
            # Show Stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall", player_data['overall'])
            c2.metric("Pace", player_data['pace'])
            c3.metric("Shooting", player_data['shooting'])
            
            # Find Similar Players
            st.markdown(f"""
            <div class="result-box">
                <b>🤝 Recommended Alternatives (Same Cluster):</b><br>
                If you cannot afford <b>{player_name}</b>, try these players who share the same playing style:
            </div>
            """, unsafe_allow_html=True)
            
            # Filter for same cluster but not the same player
            same_cluster_df = df[(df['cluster'] == cluster_id) & (df['short_name'] != player_name)]
            
            if not same_cluster_df.empty:
                recommendations = same_cluster_df.sample(min(5, len(same_cluster_df)))
                st.table(recommendations[['short_name', 'overall', 'pace', 'shooting', 'defending']])
            else:
                st.warning("No similar players found.")

    # === SECTION 4: ABOUT ===
    elif menu == "ℹ️ About Me":
        st.title("ℹ️ About the Developer")
        st.write("Name: **Kyaw Toe Toe Han**")
        st.write("Student ID: **PIUS20230059**")
        st.write("Project: **Final Term (Unsupervised Learning)**")
        
        st.success("This app demonstrates K-Means Clustering, Feature Scaling, and Interactive Deployment.")

# --- 5. RUN LOGIC ---
if st.session_state['page'] == 'welcome':
    show_welcome()
else:
    show_main_app()
