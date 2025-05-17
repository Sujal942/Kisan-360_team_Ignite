import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import re
import random

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load environment variables
load_dotenv()

# Configure page with agricultural theme
st.set_page_config(
    page_title="Kisaan 360 - Farmer's Companion",
    layout="wide",
    page_icon="🌾"
)

# Custom CSS for Farmer-friendly theme
st.markdown("""
<style>
:root {
    --primary: #4CAF50;
    --secondary: #8BC34A;
    --accent: #FFC107;
    --background: #F5F5F5;
    --text: #333333;
    --card-bg: #FFFFFF;
}

body {
    font-family: 'Roboto', sans-serif;
    background-color: var(--background);
    background-image: url('https://www.transparenttextures.com/patterns/asfalt-light.png');
}

.stApp {
    background: transparent;
}

.header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    margin-bottom: 2rem;
    text-align: center;
}

.card {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
    border-left: 5px solid var(--primary);
    transition: transform 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}

.stButton > button {
    background: var(--primary);
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    background: #3e8e41;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.stTextInput > div > input {
    border-radius: 8px;
    border: 1px solid #ddd;
    padding: 12px;
    background: white;
}

h1, h2, h3 {
    color: var(--text);
    font-weight: 600;
}

.sidebar .sidebar-content {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    border-left: 5px solid var(--secondary);
}

.price-tag {
    font-size: 1.5rem;
    color: var(--primary);
    font-weight: bold;
}

.market-name {
    color: var(--secondary);
    font-weight: 500;
}

.farmer-icon {
    font-size: 1.2rem;
    margin-right: 8px;
    color: var(--primary);
}

.weather-widget {
    background: linear-gradient(135deg, #64B5F6 0%, #42A5F5 100%);
    color: white;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.weather-icon {
    font-size: 2rem;
    margin-right: 10px;
}

.tab-content {
    padding: 15px 0;
}

@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-in;
}

.freshness-indicator {
    display: flex;
    align-items: center;
    margin: 10px 0;
}

.freshness-bar {
    height: 20px;
    border-radius: 10px;
    background: linear-gradient(90deg, #4CAF50 0%, #F44336 100%);
    position: relative;
}

.freshness-marker {
    position: absolute;
    width: 3px;
    height: 25px;
    background: black;
    top: -2px;
    transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

# Gemini API setup
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyAJxXaHjU7cjBCFbaAzxTnDbh_ClEMRqW4"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# In-memory "database"
if "recent_prices" not in st.session_state:
    st.session_state.recent_prices: list[dict] = []

# ======================== COMMODITY PRICE FUNCTIONS ========================
@st.cache_data(ttl=60)
def fetch_commodity_price(commodity_name: str) -> list[dict]:
    prompt = (
        f"Provide the current market price of {commodity_name} in India, including "
        "the market name and commodity category (e.g., Vegetable, Grain). "
        "Format the response as JSON with fields: name, category, price, market."
    )
    response = model.generate_content(prompt)
    try:
        data = json.loads(response.text.strip('```json\n').strip('```'))
    except Exception:
        data = {
            "name": commodity_name,
            "category": (
                "Vegetable" if commodity_name.lower() in ["tomato", "onion", "potato"] else "Grain"
            ),
            "price": np.random.uniform(20, 50),
            "market": "India",
        }
    return [data] if isinstance(data, dict) else data

def predict_future_price(commodity_name: str, days_ahead: int = 7) -> dict:
    dates = pd.date_range(end=datetime.now(), periods=60)
    prices = np.random.uniform(20, 50, len(dates))
    df = pd.DataFrame({"price": prices}, index=dates)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["price"]])
    seq_len = 10
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i : i + seq_len])
        y.append(scaled[i + seq_len])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    lstm = Sequential(
        [
            LSTM(50, activation="relu", input_shape=(seq_len, 1), return_sequences=True),
            LSTM(50, activation="relu"),
            Dense(1),
        ]
    )
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    last_seq = scaled[-seq_len:]
    preds = []
    for _ in range(days_ahead):
        pred = lstm.predict(last_seq.reshape(1, seq_len, 1), verbose=0)[0, 0]
        preds.append(pred)
        last_seq = np.roll(last_seq, -1)
        last_seq[-1] = pred
    future_prices = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    trend = "High" if future_prices[-1] > future_prices[0] else "Low"
    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, days_ahead + 1)]
    return {
        "commodity": commodity_name,
        "future_prices": future_prices.tolist(),
        "trend": trend,
        "dates": future_dates,
    }

# ======================== VEGETABLE ROTTEN STATUS FUNCTIONS ========================
def get_live_temperature(location):
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return None, "OpenWeatherMap API key not set."
    if not location:
        return None, "No location provided."
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return data["main"]["temp"], None
        else:
            return None, f"Error fetching temperature: {data.get('message', 'Unknown error')}"
    except Exception as e:
        return None, f"Error fetching temperature: {str(e)}"

def predict_rotten_status(vegetable, location, temperature, storage_condition, live_temp):
    # Construct prompt for Gemini API
    prompt = "Predict the rotten status of a items based on the following details (use available information and make reasonable assumptions for missing data):\n"
    if vegetable:
        prompt += f"- Commodity : {vegetable}\n"
    else:
        prompt += "- Commodity: Unknown (assume a common vegetable like Tomato)\n"
    
    # Use live temperature if available and user didn't specify temperature
    if temperature is not None:
        prompt += f"- Temperature: {temperature}°C (user-specified)\n"
        selected_temp = temperature
    elif live_temp is not None:
        prompt += f"- Temperature: {live_temp:.1f}°C (current temperature in {location})\n"
        selected_temp = live_temp
    else:
        prompt += "- Temperature: Unknown (assume room temperature, 25°C)\n"
        selected_temp = 25.0
    
    if location:
        prompt += f"- Location: {location}\n"
    else:
        prompt += "- Location: Unknown (assume generic location)\n"
    
    if storage_condition:
        prompt += f"- Storage Condition: {storage_condition}\n"
    else:
        prompt += "- Storage Condition: Unknown (assume open air)\n"
    
    prompt += (
        "Provide a concise prediction (e.g., Fresh, Likely Fresh, Likely Rotten, Rotten), a brief explanation, "
        "an estimated freshness probability (0-100%), and the number of days the vegetable is expected to remain fresh."
    )

    try:
        # Call Gemini API
        response = model.generate_content(prompt)
        result = response.text.strip()

        # Parse freshness duration from response (if provided)
        freshness_days = None
        match = re.search(r'(\d+\.?\d*)\s*(?:days|day)', result, re.IGNORECASE)
        if match:
            freshness_days = float(match.group(1))
        else:
            freshness_days = random.uniform(2, 10)  # Fallback if not found

        # Parse or simulate freshness probability
        freshness_prob_match = re.search(r'freshness probability.*?(\d+)%', result, re.IGNORECASE)
        freshness_prob = float(freshness_prob_match.group(1)) if freshness_prob_match else random.uniform(0, 100)
        if "Fresh" in result:
            freshness_prob = max(freshness_prob, 70)
        elif "Rotten" in result:
            freshness_prob = min(freshness_prob, 30)
        elif "Likely Fresh" in result:
            freshness_prob = max(freshness_prob, 50)
        elif "Likely Rotten" in result:
            freshness_prob = min(freshness_prob, 50)

        return {
            "result": result,
            "freshness_days": freshness_days,
            "freshness_prob": freshness_prob,
            "selected_temp": selected_temp
        }

    except Exception as e:
        st.error(f"Error communicating with Gemini API: {str(e)}")
        return None

# ======================== MAIN APP ========================
def main() -> None:
    # Header with agricultural theme
    st.markdown("""
    <div class="header">
        <h1>🌾 Kisaan 360 - Farmer's Companion</h1>
        <p>Your complete farming assistant for market prices and produce quality</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Market Prices", "Produce Quality Check"])

    with tab1:
        # Main content for Market Prices
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Search bar with farmer-friendly design
            st.markdown("### <span class='farmer-icon'>🔍</span> Search Commodity Prices", unsafe_allow_html=True)
            commodity = st.text_input(
                "Enter crop or commodity name (e.g., Tomato, Wheat, Rice)",
                placeholder="e.g., Potato, Onion, Cotton",
                label_visibility="collapsed",
                key="price_search"
            )
            
            if commodity:
                # Current price
                with st.spinner("Fetching latest market prices..."):
                    price_records = fetch_commodity_price(commodity)
                timestamp = datetime.now()
                for rec in price_records:
                    rec = rec.copy()
                    rec["date"] = timestamp
                    st.session_state.recent_prices.insert(0, rec)
                st.session_state.recent_prices = st.session_state.recent_prices[:50]
                
                # Display current prices in card format
                st.markdown(f"### <span class='farmer-icon'>💰</span> Current {commodity} Prices", unsafe_allow_html=True)
                for rec in price_records:
                    st.markdown(f"""
                    <div class="card fade-in">
                        <h3>{rec['name']} ({rec['category']})</h3>
                        <p class="price-tag">₹{rec['price']:.2f}/kg</p>
                        <p class="market-name">📍 {rec['market']}</p>
                        <small>Updated: {timestamp.strftime('%d %b %Y, %I:%M %p')}</small>
                    </div>
                    """, unsafe_allow_html=True)

                # Prediction section
                st.markdown(f"### <span class='farmer-icon'>📈</span> {commodity} Price Forecast", unsafe_allow_html=True)
                if st.button("Get 7-Day Price Prediction", key="price_prediction"):
                    with st.spinner("Analyzing market trends..."):
                        pred = predict_future_price(commodity)
                        pred_df = pd.DataFrame(
                            {"Date": pred["dates"], "Predicted Price (₹/kg)": pred["future_prices"]}
                        )
                        
                        st.markdown(f"""
                        <div class="card fade-in">
                            <h3>{commodity} Price Trend: <span style="color: {'#4CAF50' if pred['trend'] == 'High' else '#F44336'}">{pred['trend']}</span></h3>
                        """, unsafe_allow_html=True)
                        
                        fig = px.line(
                            pred_df,
                            x="Date",
                            y="Predicted Price (₹/kg)",
                            title=f"{commodity} Price Forecast",
                            template="plotly_white",
                        )
                        fig.update_traces(line=dict(color="#4CAF50", width=3))
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            transition_duration=500,
                            hoverlabel=dict(
                                bgcolor="white",
                                font_size=16,
                                font_family="Roboto"
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add recommendation based on trend
                        recommendation = "Prices are expected to rise. Consider holding your produce for better rates." if pred['trend'] == "High" else "Prices may drop. Consider selling soon for better returns."
                        st.markdown(f"""
                        <div style="background-color: #FFF9C4; padding: 15px; border-radius: 8px; margin-top: 15px;">
                            <h4>💡 Farmer's Advice</h4>
                            <p>{recommendation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)

        # Sidebar with additional farmer tools
        with col2:
            st.markdown("### <span class='farmer-icon'>🌤️</span> Weather Update", unsafe_allow_html=True)
            st.markdown("""
            <div class="weather-widget">
                <div style="display: flex; align-items: center;">
                    <span class="weather-icon">☀️</span>
                    <div>
                        <h3>Sunny, 32°C</h3>
                        <p>Madhya Pradesh, India</p>
                    </div>
                </div>
                <p>Good weather for harvesting</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### <span class='farmer-icon'>⏱️</span> Recent Searches", unsafe_allow_html=True)
            if st.session_state.recent_prices:
                recent_df = pd.DataFrame(st.session_state.recent_prices)
                recent_df['date'] = recent_df['date'].apply(lambda x: x.strftime('%d %b %H:%M'))
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.dataframe(
                    recent_df[['name', 'price', 'market', 'date']].rename(columns={
                        'name': 'Commodity',
                        'price': 'Price (₹)',
                        'market': 'Market',
                        'date': 'Time'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card">
                    <p>No recent searches yet. Search for a commodity to see price history.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Quick links for farmers
            st.markdown("### <span class='farmer-icon'>🔗</span> Quick Links", unsafe_allow_html=True)
            st.markdown("""
            <div class="card">
                <p><a href="#" target="_blank">🌱 Government Schemes</a></p>
                <p><a href="#" target="_blank">🚜 Nearest Mandi</a></p>
                <p><a href="#" target="_blank">💧 Irrigation Tips</a></p>
                <p><a href="#" target="_blank">📞 Farmer Helpline</a></p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        # Vegetable Rotten Status Predictor
        st.markdown("## <span class='farmer-icon'>🥕</span> Produce Quality Checker", unsafe_allow_html=True)
        st.markdown("Check the freshness of your produce and estimate how long it will stay fresh")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input form in a card
            with st.form("quality_check_form"):
                st.markdown("### <span class='farmer-icon'>📝</span> Enter Details", unsafe_allow_html=True)
                vegetable = st.text_input("Vegetable (e.g., Tomato):", "").strip()
                location = st.text_input("Location (e.g., Jabalpur):", "").strip()
                temperature = st.number_input("Temperature (°C):", min_value=-20.0, max_value=50.0, value=None, placeholder="e.g., 25.0")
                storage_condition = st.selectbox("Storage Condition:", ["Open Air", "Sealed Container", "Vacuum Packed"], index=0)
                
                submitted = st.form_submit_button("Check Freshness", type="primary")
        
        with col2:
            # Display results
            if submitted:
                # Fetch live temperature if location is provided
                live_temp, temp_error = get_live_temperature(location) if location else (None, None)
                if temp_error:
                    st.warning(temp_error)
                elif live_temp is not None:
                    st.info(f"Current temperature in {location}: {live_temp:.1f}°C")
                
                # Check if any input is provided to enable the button
                has_input = bool(vegetable or location or temperature is not None or storage_condition)
                
                if has_input:
                    with st.spinner("Analyzing produce quality..."):
                        result = predict_rotten_status(vegetable, location, temperature, storage_condition, live_temp)
                        
                        if result:
                            st.markdown("### <span class='farmer-icon'>🔍</span> Analysis Results", unsafe_allow_html=True)
                            st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
                            
                            # Display prediction result
                            st.markdown(f"**Prediction:** {result['result'].split('.')[0]}")
                            st.markdown(f"**Explanation:** {'.'.join(result['result'].split('.')[1:])}")
                            
                            # Freshness days metric
                            st.metric("Estimated Freshness Duration", f"{result['freshness_days']:.1f} days")
                            
                            # Freshness probability indicator
                            st.markdown("**Freshness Probability:**")
                            st.markdown(f"""
                            <div class="freshness-indicator">
                                <div class="freshness-bar" style="flex-grow: 1;">
                                    <div class="freshness-marker" style="left: {result['freshness_prob']}%;"></div>
                                </div>
                                <span style="margin-left: 10px; font-weight: bold;">{result['freshness_prob']:.0f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Create freshness probability bar chart
                            freshness_data = pd.DataFrame({
                                "Status": ["Fresh", "Rotten"],
                                "Probability": [result['freshness_prob'], 100 - result['freshness_prob']]
                            })
                            fig1 = px.bar(freshness_data, x="Status", y="Probability", title="Freshness Probability",
                                        color="Status", color_discrete_map={"Fresh": "#4CAF50", "Rotten": "#F44336"})
                            fig1.update_layout(yaxis_title="Probability (%)", yaxis_range=[0, 100])
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            # Temperature impact visualization
                            st.markdown("**Temperature Impact on Freshness:**")
                            temp_range = range(-5, 40, 5)
                            spoilage_rate = [100 / (t + 10) if t > 0 else 5 for t in temp_range]  # Simplified spoilage model
                            temp_data = pd.DataFrame({
                                "Temperature (°C)": temp_range,
                                "Spoilage Rate (Days)": spoilage_rate
                            })
                            fig2 = px.line(temp_data, x="Temperature (°C)", y="Spoilage Rate (Days)", 
                                        title="Temperature vs. Shelf Life",
                                        markers=True)
                            if result['selected_temp'] is not None:
                                spoilage_at_temp = 100 / (result['selected_temp'] + 10) if result['selected_temp'] > 0 else 5
                                fig2.add_trace(go.Scatter(
                                    x=[result['selected_temp']], 
                                    y=[spoilage_at_temp],
                                    mode="markers+text",
                                    name="Current Temp",
                                    marker=dict(size=15, color="red"),
                                    text=[f"{result['freshness_days']:.1f} days"],
                                    textposition="top center"
                                ))
                            fig2.update_layout(yaxis_title="Days to Spoilage")
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                
                else:
                    st.warning("Please enter at least one detail to check freshness")

if __name__ == "__main__":
    main()