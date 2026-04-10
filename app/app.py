import streamlit as st
import pandas as pd
import sqlite3
import pickle
import os

# ----------------------------
# PAGE CONFIG (MUST BE FIRST)
# ----------------------------
st.set_page_config(page_title="Creator Intelligence Dashboard", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.block-container {
    padding-top: 1rem;
}

.card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.metric-card {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h2>👋 Good Day, Creator!</h2>
    <p>Here’s your content performance overview</p>
</div>
""", unsafe_allow_html=True)



st.title("📊 Creator Intelligence Dashboard")

# ----------------------------
# PATH SETUP (ROBUST & PRODUCTION-READY)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
db_path = os.path.join(BASE_DIR, 'database', 'db.sqlite3')
model_path = os.path.join(BASE_DIR, 'model.pkl')

# Validate required files exist
def validate_paths():
    """Validate critical files before proceeding"""
    errors = []
    if not os.path.exists(db_path):
        errors.append(f"Database not found: {db_path}")
    if not os.path.exists(model_path):
        errors.append(f"Model not found: {model_path}")
    return errors

path_errors = validate_paths()
if path_errors:
    st.error("Configuration Error - Missing Files:")
    for error in path_errors:
        st.error(f"  • {error}")
    st.stop()

# Load database with error handling
try:
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM content_metrics"
    df = pd.read_sql(query, conn)
    conn.close()
except Exception as e:
    st.error(f"Failed to load database: {str(e)}")
    st.stop()

# Load model with error handling
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()


# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.markdown("## 🚀 CreatorIQ")
st.sidebar.markdown("Optimize your content strategy")
st.sidebar.markdown("---")

# Add actual filters if data exists
if not df.empty and 'platform' in df.columns:
    platform_filter = st.sidebar.multiselect(
        "Select Platform",
        options=df['platform'].unique(),
        default=df['platform'].unique()
    )
else:
    platform_filter = []

if not df.empty and 'content_type' in df.columns:
    content_filter = st.sidebar.multiselect(
        "Select Content Type",
        options=df['content_type'].unique(),
        default=df['content_type'].unique()
    )
else:
    content_filter = []

# Apply filters
if not df.empty and platform_filter and content_filter:
    df_filtered = df[
        (df['platform'].isin(platform_filter)) &
        (df['content_type'].isin(content_filter))
    ]
else:
    df_filtered = df

st.sidebar.markdown("---")
st.sidebar.markdown("👤 Profile: Creator")




# ----------------------------
# CREATE TABS
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analytics", "🧠 Insights"])

# ----------------------------
# TAB 1 — OVERVIEW (KPIs)
# ----------------------------
with tab1:
    st.subheader("📊 Overview")

    col1, col2, col3, col4 = st.columns(4)

    def metric_card(title, value):
        return f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <h2>{value}</h2>
        </div>
        """
    
    # Display metrics with error handling
    if not df_filtered.empty:
        if 'views' in df_filtered.columns:
            col1.markdown(metric_card("Views", f"{int(df_filtered['views'].sum()):,}"), unsafe_allow_html=True)
        if 'engagement_rate' in df_filtered.columns:
            col2.markdown(metric_card("Engagement", round(df_filtered['engagement_rate'].mean(), 3)), unsafe_allow_html=True)
        if 'followers_gained' in df_filtered.columns:
            col3.markdown(metric_card("Followers", f"{int(df_filtered['followers_gained'].sum()):,}"), unsafe_allow_html=True)
        if 'virality_score' in df_filtered.columns:
            col4.markdown(metric_card("Virality", round(df_filtered['virality_score'].mean(), 3)), unsafe_allow_html=True)
    else:
        st.warning("⚠️ No data available. Please ensure the database is populated and filters are correctly set.")


# ----------------------------
# TAB 2 — ANALYTICS
# ----------------------------
with tab2:
    st.subheader("📈 Performance Trends")

    if not df_filtered.empty:
        import plotly.express as px

        # Prepare data
        if 'date' in df_filtered.columns and 'views' in df_filtered.columns:
            views_trend = df_filtered.groupby('date')['views'].sum().reset_index()
            fig_views = px.line(views_trend, x='date', y='views', title='Views Over Time')
        else:
            fig_views = None

        if 'content_type' in df_filtered.columns and 'engagement_rate' in df_filtered.columns:
            engagement_ct = df_filtered.groupby('content_type')['engagement_rate'].mean().reset_index()
            fig_engagement = px.bar(engagement_ct, x='content_type', y='engagement_rate', title='Engagement by Content Type')
        else:
            fig_engagement = None

        if 'platform' in df_filtered.columns and 'engagement_rate' in df_filtered.columns:
            platform_perf = df_filtered.groupby('platform')['engagement_rate'].mean().reset_index()
            fig_platform = px.bar(platform_perf, x='platform', y='engagement_rate', title='Platform Comparison')
        else:
            fig_platform = None

        # ----------------------------
        # GRID LAYOUT
        # ----------------------------
        col1, col2 = st.columns([2, 1])

        with col1:
            if fig_views:
                st.plotly_chart(fig_views, use_container_width=True)

        with col2:
            st.markdown("### 📊 Quick Insights")
            if 'content_type' in df_filtered.columns and 'engagement_rate' in df_filtered.columns and not df_filtered.empty:
                best_content = df_filtered.groupby('content_type')['engagement_rate'].mean().idxmax()
                st.write(f"🔥 Best Content: **{best_content}**")

        # SECOND ROW
        col3, col4 = st.columns(2)

        with col3:
            if fig_engagement:
                st.plotly_chart(fig_engagement, use_container_width=True)

        with col4:
            if fig_platform:
                st.plotly_chart(fig_platform, use_container_width=True)
    else:
        st.warning("⚠️ No data available for analytics.")


    
# ----------------------------
# TAB 3 — INSIGHTS + RECOMMENDER + PREDICTION
# ----------------------------
with tab3:
    st.subheader("🧠 Smart Insights")

    if not df_filtered.empty and 'content_type' in df_filtered.columns and 'engagement_rate' in df_filtered.columns:
        content_perf = df_filtered.groupby('content_type')['engagement_rate'].mean()
        
        if 'day_of_week' in df_filtered.columns:
            day_perf = df_filtered.groupby('day_of_week')['engagement_rate'].mean()
            best_day = day_perf.idxmax()
        else:
            best_day = "N/A"

        best_content = content_perf.idxmax()
        worst_content = content_perf.idxmin()

        st.success(f"🔥 Best Content Type: {best_content}")
        st.warning(f"⚠️ Underperforming Content: {worst_content}")
        st.info(f"📅 Best Day to Post: {best_day}")

        # ----------------------------
        # TIME ANALYSIS
        # ----------------------------
        if 'date' in df_filtered.columns:
            df_filtered_copy = df_filtered.copy()
            df_filtered_copy['date'] = pd.to_datetime(df_filtered_copy['date'])
            df_filtered_copy['hour'] = df_filtered_copy['date'].dt.hour

            hour_perf = df_filtered_copy.groupby('hour')['engagement_rate'].mean()
            best_hour = hour_perf.idxmax()
            st.write(f"⏰ Best Posting Hour: **{best_hour}:00**")
        else:
            best_hour = "N/A"

        # ----------------------------
        # RECOMMENDATION ENGINE
        # ----------------------------
        st.subheader("🚀 What Should You Post Next?")

        st.success(f"""
        📌 Based on your analytics:

        👉 Post a **{best_content}**  
        👉 On **{best_day}**  
        👉 Around **{best_hour}:00** (if available)

        💡 Strategy:
        Focus on high-engagement formats and avoid {worst_content} unless optimized.
        """)

        # ----------------------------
        # TOP POSTS
        # ----------------------------
        st.subheader("🏆 Top Performing Content")

        top_posts = df_filtered.sort_values(by='engagement_rate', ascending=False).head(5)
        st.dataframe(top_posts[['platform', 'content_type', 'views', 'engagement_rate']])
        
        # ----------------------------
        # 🔮 PREDICTION SECTION
        # ----------------------------
        st.subheader("🔮 Predict Engagement")

        views = st.number_input("Views", value=10000)
        likes = st.number_input("Likes", value=1000)
        comments = st.number_input("Comments", value=100)
        shares = st.number_input("Shares", value=50)
        watch_time = st.number_input("Watch Time", value=5000)
        followers = st.number_input("Followers Gained", value=200)

        if st.button("Predict Engagement"):
            try:
                input_data = pd.DataFrame([{
                    'views': views,
                    'likes': likes,
                    'comments': comments,
                    'shares': shares,
                    'watch_time': watch_time,
                    'followers_gained': followers
                }])

                prediction = model.predict(input_data)
                st.success(f"📊 Predicted Engagement Rate: {round(prediction[0], 4)}")
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
    else:
        st.warning("⚠️ No data available for insights.")


