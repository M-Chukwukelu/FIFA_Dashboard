import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="FIFA Wonderkid Predictor", layout="centered")

# Load Models
@st.cache_resource
def load_models():
    attacker_model = tf.keras.models.load_model('attacker_growth_predictor.keras')
    midfielder_model = tf.keras.models.load_model('midfielder_growth_predictor.keras')
    defender_model = tf.keras.models.load_model('defender_growth_predictor.keras')
    goalkeeper_model = tf.keras.models.load_model('goalkeeper_growth_predictor.keras')
    return attacker_model, midfielder_model, defender_model, goalkeeper_model

attacker_model, midfielder_model, defender_model, goalkeeper_model = load_models()

# Wonderkids data
@st.cache_data
def load_wonderkid_data():
    df = pd.read_csv("wonderkids_compact.csv")
    return df

wonderkids_df = load_wonderkid_data()

# Sidebar Navigation
page = st.sidebar.radio("Choose a mode", ["Predict Growth", "Wonderkid Analysis"])
st.sidebar.markdown("---")
st.sidebar.write(" Built with TensorFlow & Streamlit")

# Common Feature Maps
attacker_features = [
    'age', 'potential', 'pace', 'shooting', 'passing', 'dribbling', 'physic',
    'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
    'attacking_short_passing', 'attacking_volleys', 'mentality_positioning',
    'mentality_vision', 'mentality_penalties', 'mentality_composure']

midfielder_features = [
    'age', 'potential', 'pace', 'shooting', 'passing', 'dribbling', 'physic',
    'attacking_short_passing', 'mentality_vision', 'mentality_positioning',
    'skill_long_passing', 'mentality_composure']

defender_features = [
    'age', 'potential', 'pace', 'defending', 'physic', 'movement_reactions',
    'mentality_aggression', 'mentality_interceptions', 'defending_marking_awareness',
    'defending_standing_tackle', 'defending_sliding_tackle', 'mentality_composure']

goalkeeper_features = [
    'age', 'potential', 'goalkeeping_diving', 'goalkeeping_handling',
    'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes',
    'goalkeeping_speed']

model_map = {
    "Attacker": attacker_model,
    "Midfielder": midfielder_model,
    "Defender": defender_model,
    "Goalkeeper": goalkeeper_model
}

feature_map = {
    "Attacker": attacker_features,
    "Midfielder": midfielder_features,
    "Defender": defender_features,
    "Goalkeeper": goalkeeper_features
}

# Mode: Prediction Tool
if page == "Predict Growth":
    st.markdown("<h1 style='text-align:center;color:#e8dcff;'>FIFA Career Mode Growth Predictor</h1>", unsafe_allow_html=True)
    position = st.selectbox("Select Player Position", ("Attacker", "Midfielder", "Defender", "Goalkeeper"))
    st.markdown(f"<h3 style='color:#beaae6;'>Enter Player Stats ({position})</h3>", unsafe_allow_html=True)

    age = st.slider('Age', 15, 45, 20)
    overall = st.slider('Current Overall Rating', 40, 99, 65)

    input_features = []
    model = model_map[position]

    if position == "Attacker":
        potential = st.slider("Potential", 40, 99, 80)
        pace = st.slider("Pace", 0, 100, 70)
        shooting = st.slider("Shooting", 0, 100, 65)
        passing = st.slider("Passing", 0, 100, 60)
        dribbling = st.slider("Dribbling", 0, 100, 68)
        physic = st.slider("Physical", 0, 100, 55)
        attacking_crossing = st.slider("Crossing", 0, 100, 60)
        attacking_finishing = st.slider("Finishing", 0, 100, 70)
        attacking_heading_accuracy = st.slider("Heading Accuracy", 0, 100, 65)
        attacking_short_passing = st.slider("Short Passing", 0, 100, 68)
        attacking_volleys = st.slider("Volleys", 0, 100, 60)
        mentality_positioning = st.slider("Positioning", 0, 100, 65)
        mentality_vision = st.slider("Vision", 0, 100, 64)
        mentality_penalties = st.slider("Penalties", 0, 100, 55)
        mentality_composure = st.slider("Composure", 0, 100, 63)

        input_features = np.array([[age, potential, pace, shooting, passing, dribbling, physic,
                                    attacking_crossing, attacking_finishing, attacking_heading_accuracy,
                                    attacking_short_passing, attacking_volleys, mentality_positioning,
                                    mentality_vision, mentality_penalties, mentality_composure]])

    elif position == "Midfielder":
        pace = st.slider('Pace', 0, 100, 65)
        shooting = st.slider('Shooting', 0, 100, 60)
        passing = st.slider('Passing', 0, 100, 70)
        dribbling = st.slider('Dribbling', 0, 100, 68)
        physic = st.slider('Physical', 0, 100, 62)
        short_passing = st.slider('Short Passing', 0, 100, 72)
        vision = st.slider('Vision', 0, 100, 65)
        positioning = st.slider('Positioning', 0, 100, 60)
        long_passing = st.slider('Long Passing', 0, 100, 68)
        composure = st.slider('Composure', 0, 100, 66)

        input_features = np.array([[age, overall, pace, shooting, passing, dribbling, physic,
                                    short_passing, vision, positioning, long_passing, composure]])

    elif position == "Defender":
        pace = st.slider('Pace', 0, 100, 60)
        passing = st.slider('Passing', 0, 100, 58)
        physic = st.slider('Physical', 0, 100, 70)
        defending = st.slider('Defending', 0, 100, 75)
        standing_tackle = st.slider('Standing Tackle', 0, 100, 75)
        marking = st.slider('Marking Awareness', 0, 100, 72)
        strength = st.slider('Strength', 0, 100, 78)
        jumping = st.slider('Jumping', 0, 100, 70)
        heading = st.slider('Heading Accuracy', 0, 100, 65)

        input_features = np.array([[age, overall, pace, passing, physic,
                                    defending, standing_tackle, marking, strength, jumping, heading]])

    else:  # Goalkeeper
        gk_diving = st.slider('GK Diving', 0, 100, 65)
        gk_handling = st.slider('GK Handling', 0, 100, 63)
        gk_kicking = st.slider('GK Kicking', 0, 100, 60)
        gk_positioning = st.slider('GK Positioning', 0, 100, 62)
        gk_reflexes = st.slider('GK Reflexes', 0, 100, 68)

        input_features = np.array([[age, overall, gk_diving, gk_handling, gk_kicking, gk_positioning, gk_reflexes]])

    if st.button('Predict Growth'):
        predicted_growth = model.predict(input_features)[0][0]
        final_rating = overall + predicted_growth
        st.success(f"Predicted Growth: {predicted_growth:.2f} points")
        st.success(f"Predicted Final Rating: {final_rating:.2f}")
        if predicted_growth >= 8 and age <= 20:
            st.balloons()
            st.markdown("<b>Wonderkid Potential Detected!</b>", unsafe_allow_html=True)

# Mode: Wonderkid Analysis
elif page == "Wonderkid Analysis":
    lookup = wonderkids_df[['sofifa_id','short_name']].drop_duplicates()

    # let the user pick by name, but under the hood carry the ID
    selected = st.selectbox(
        "Select a Wonderkid",
        lookup.itertuples(index=False),
        format_func=lambda row: row.short_name
    ).sofifa_id

    # now filter cleanly
    df_p = wonderkids_df[ wonderkids_df['sofifa_id'] == selected ] \
            .sort_values('fifa_year') \
            .reset_index(drop=True)

    role  = df_p['position_group'].iloc[0]
    feats = feature_map[role]
    mdl   = model_map[role]

    # precompute global means for fill-in
    global_means = wonderkids_df[feats].mean()

    bursts = []
    for base_year in df_p['fifa_year']:
        target_year = base_year + 3
        if target_year not in df_p['fifa_year'].values:
            continue

        base_row = df_p.loc[df_p['fifa_year']==base_year].iloc[0]
        X = base_row[feats]


        x_input = X.values.astype(float).reshape(1, -1)
        pred_growth = mdl.predict(x_input, verbose=0)[0][0]
        pred_final  = base_row['overall'] + pred_growth

        actual_final = df_p.loc[df_p['fifa_year']==target_year, 'overall'].iloc[0]
        bursts.append((base_year, target_year, pred_final, actual_final))

    if bursts:
        years_from, years_to, predicted, actuals = zip(*bursts)
        plt.figure(figsize=(10,5))
        plt.plot(years_to, predicted,   '-o', label='Predicted Final')
        plt.plot(years_to, actuals,     '-x', label='Actual Overall')
        player_name = df_p['wonderkid_name'].iloc[0]
        plt.title(f"{player_name}: {years_from[0]}→{years_to[-1]} 3-Year Bursts")
        plt.xlabel("Target FIFA Year")
        plt.ylabel("Rating")
        plt.legend()
        st.pyplot(plt.gcf())

        # simple summary
        diffs = [p - a for p,a in zip(predicted, actuals)]
        avg_diff = sum(diffs)/len(diffs)
        if avg_diff < -3:
            verdict = "On average, fell short of expectations."
        elif avg_diff < 3:
            verdict = "On average, met expectations."
        else:
            verdict = "On average, exceeded expectations."

        st.markdown(f"### Conclusion: {verdict}")
    else:
        st.warning("Not enough 3-year span data for this player.")
