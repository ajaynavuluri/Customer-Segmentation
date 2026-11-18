
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path
from datetime import datetime

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "segmentation_model.pkl"
DATA_PATH = BASE_DIR / "customers_with_segments.csv"


# =====================================================
# 2. PROFESSIONAL UI DESIGN
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f4f7fc;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background-color: #14213d;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    background: linear-gradient(
        115deg,
        #14213d,
        #2455a4,
        #3478d4
    );
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #e2ecff;
    font-size: 14px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e6ebf3;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(16,35,70,0.04);
}

div[data-testid="stMetricLabel"] {
    color: #667085;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #14213d;
    font-weight: 800;
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #e6ebf3;
    border-radius: 16px;
    padding: 20px;
}

div.stButton > button,
div.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
}

</style>

<div class="hero">
    <h1>📊 Customer Segmentation Analytics</h1>
    <p>
    Understand customer behaviour, explore customer groups,
    and predict the segment of a new customer.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD MODEL AND DATASET
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


if not MODEL_PATH.exists():
    st.error("segmentation_model.pkl file not found!")
    st.info("Keep the model file in the same folder as app.py.")
    st.stop()

if not DATA_PATH.exists():
    st.error("customers_with_segments.csv file not found!")
    st.info("Keep the CSV dataset in the same folder as app.py.")
    st.stop()

try:
    model = load_model()
    data = load_data()

except Exception as error:
    st.error("Unable to load model or dataset.")
    st.exception(error)
    st.stop()


# Validate model structure

required_keys = [
    "scaler",
    "pca",
    "kmeans",
    "features",
    "names"
]

missing_keys = [
    key for key in required_keys
    if key not in model
]

if missing_keys:
    st.error(f"Missing model items: {missing_keys}")
    st.stop()


required_columns = [
    "Segment",
    "Total_Spent",
    "Income",
    "Total_Purchases"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()


FEATURES = list(model["features"])


# =====================================================
# 4. SESSION STATE FOR PREDICTION HISTORY
# =====================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("📊 Segmentation")
st.sidebar.caption("Customer Intelligence Dashboard")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Segment Analysis",
        "Predict Customer"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Model Workflow")

st.sidebar.caption(
    "Preprocessed Data → Scaling → PCA → K-Means"
)

st.sidebar.caption("3 Customer Segments")

st.sidebar.divider()

st.sidebar.caption(
    "Customer Segmentation Project"
)


# =====================================================
# 6. HELPER FUNCTIONS
# =====================================================

def format_money(value):
    return f"₹{value:,.0f}"


def get_segment_name(cluster_id):

    names = model["names"]

    if isinstance(names, dict):
        return names.get(
            cluster_id,
            names.get(str(cluster_id), f"Cluster {cluster_id}")
        )

    return names[cluster_id]


def get_recommendation(segment):

    name = str(segment).lower()

    if "premium" in name:

        return (
            "Consider testing loyalty rewards, premium bundles "
            "and early access to new products. Measure repeat "
            "purchases before expanding the campaign."
        )

    elif "deal" in name:

        return (
            "Consider testing limited-time offers, coupons "
            "and value bundles. Track whether promotions "
            "increase profitable purchases."
        )

    elif "budget" in name:

        return (
            "Consider testing affordable bundles and clear "
            "value messaging. Compare conversion rates "
            "and average order value."
        )

    return (
        "Study this customer group's spending and purchase "
        "patterns before selecting a marketing strategy."
    )


# =====================================================
# 7. OVERVIEW PAGE
# =====================================================

if page == "Overview":

    st.subheader("Business Overview")

    st.caption(
        "A quick snapshot of customers in your prepared dataset."
    )

    total_customers = len(data)

    total_segments = data["Segment"].nunique()

    average_spend = data["Total_Spent"].mean()

    average_income = data["Income"].mean()


    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Customer Segments",
        total_segments
    )

    col3.metric(
        "Average Spend",
        format_money(average_spend)
    )

    col4.metric(
        "Average Income",
        format_money(average_income)
    )


    # CUSTOMER CHARTS

    st.markdown("### Customer Insights")

    left, right = st.columns(2)

    counts = (
        data["Segment"]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Customers")
    )

    with left:

        fig = px.pie(
            counts,
            names="Segment",
            values="Customers",
            hole=0.58,
            title="Customer Distribution",
            color_discrete_sequence=[
                "#3478d4",
                "#18a999",
                "#f4a340"
            ]
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        fig.update_layout(
            paper_bgcolor="white",
            margin=dict(t=55, b=15)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    spend = (
        data.groupby("Segment", as_index=False)["Total_Spent"]
        .mean()
        .sort_values("Total_Spent", ascending=False)
    )

    with right:

        fig = px.bar(
            spend,
            x="Segment",
            y="Total_Spent",
            color="Segment",
            title="Average Spend by Segment",
            text_auto=".0f"
        )

        fig.update_layout(
            showlegend=False,
            paper_bgcolor="white",
            margin=dict(t=55, b=15),
            xaxis_title="Customer Segment",
            yaxis_title="Average Spend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # SEGMENT SUMMARY

    st.markdown("### Segment Performance")

    summary = (
        data.groupby("Segment")
        .agg(
            Customers=("Segment", "size"),
            Average_Income=("Income", "mean"),
            Average_Spend=("Total_Spent", "mean"),
            Average_Purchases=("Total_Purchases", "mean")
        )
        .round(1)
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # PREDICTION HISTORY

    st.markdown("### Recent Customer Predictions")

    history = st.session_state.prediction_history

    if history:

        history_df = pd.DataFrame(history)

        st.caption(
            f"{len(history_df)} prediction(s) in this active session. "
            "These new customers are separate from the original dataset."
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Prediction History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No predictions yet. Go to Predict Customer "
            "to enter a new customer's details."
        )

    st.caption(
        "Original customer statistics are calculated only from "
        "the prepared dataset. Prediction history is stored "
        "separately for the active session."
    )


# =====================================================
# 8. SEGMENT ANALYSIS PAGE
# =====================================================

elif page == "Segment Analysis":

    st.subheader("Explore Customer Segments")

    st.caption(
        "Understand the spending behaviour and purchase "
        "patterns of each customer group."
    )

    segment = st.selectbox(
        "Choose Customer Segment",
        sorted(data["Segment"].dropna().unique())
    )

    segment_data = data[
        data["Segment"] == segment
    ].copy()


    # SEGMENT METRICS

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers",
        f"{len(segment_data):,}"
    )

    col2.metric(
        "Average Spend",
        format_money(segment_data["Total_Spent"].mean())
    )

    col3.metric(
        "Average Income",
        format_money(segment_data["Income"].mean())
    )


    # PRODUCT SPENDING

    st.markdown("### Product Spending Analysis")

    product_columns = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds"
    ]

    available_products = [
        col for col in product_columns
        if col in segment_data.columns
    ]

    if available_products:

        product_data = (
            segment_data[available_products]
            .mean()
            .rename_axis("Product Category")
            .reset_index(name="Average Spend")
        )

        product_data["Product Category"] = (
            product_data["Product Category"]
            .str.replace("Mnt", "", regex=False)
            .str.replace("Products", "", regex=False)
        )

        fig = px.bar(
            product_data,
            x="Product Category",
            y="Average Spend",
            color="Product Category",
            title="Average Spending by Product Category",
            text_auto=".0f"
        )

        fig.update_layout(
            showlegend=False,
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info("Product spending columns are unavailable.")


    # PURCHASE CHANNEL ANALYSIS

    st.markdown("### Purchase Channel Analysis")

    channel_columns = [
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases"
    ]

    available_channels = [
        col for col in channel_columns
        if col in segment_data.columns
    ]

    if available_channels:

        channel_data = (
            segment_data[available_channels]
            .mean()
            .rename_axis("Channel")
            .reset_index(name="Average Purchases")
        )

        channel_data["Channel"] = (
            channel_data["Channel"].replace({
                "NumWebPurchases": "Web",
                "NumCatalogPurchases": "Catalog",
                "NumStorePurchases": "Store"
            })
        )

        fig = px.bar(
            channel_data,
            x="Channel",
            y="Average Purchases",
            color="Channel",
            title="Average Purchases by Channel",
            text_auto=".1f"
        )

        fig.update_layout(
            showlegend=False,
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info("Purchase channel columns are unavailable.")


    # SUGGESTED ACTION

    st.markdown("### Suggested Marketing Action")

    st.info(get_recommendation(segment))


    # CUSTOMER TABLE

    st.markdown("### Customer Records")

    display_columns = [
        col for col in [
            "ID",
            "Age",
            "Income",
            "Total_Spent",
            "Total_Purchases",
            "Segment"
        ]
        if col in segment_data.columns
    ]

    st.dataframe(
        segment_data[display_columns].head(200),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Showing up to 200 of {len(segment_data):,} customers."
    )

    st.download_button(
        "⬇️ Download Segment CSV",
        data=segment_data.to_csv(index=False).encode("utf-8"),
        file_name="customer_segment.csv",
        mime="text/csv"
    )


# =====================================================
# 9. PREDICT CUSTOMER PAGE
# =====================================================

else:

    st.subheader("Predict a New Customer's Segment")

    st.caption(
        "Enter customer details below. The saved model will "
        "use Scaling, PCA and K-Means to predict the segment."
    )


    # CUSTOMER INPUT FORM

    with st.form("customer_prediction"):

        st.markdown("### 1. Customer Profile")

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.number_input(
                "Age (Years)",
                18, 99, 35
            )

            income = st.number_input(
                "Annual Income (₹)",
                0, 10000000, 50000,
                step=1000
            )

            children = st.number_input(
                "Number of Children",
                0, 10, 0
            )

            tenure = st.number_input(
                "Customer Tenure (Days)",
                0, 10000, 365
            )


        with col2:

            spent = st.number_input(
                "Total Product Spend (₹)",
                0, 1000000, 500,
                step=50
            )

            recency = st.number_input(
                "Days Since Last Purchase",
                0, 3650, 30
            )

            purchases = st.number_input(
                "Total Purchases",
                0, 10000, 15
            )

            deals = st.number_input(
                "Deal Purchases",
                0, 1000, 2
            )


        with col3:

            visits = st.number_input(
                "Website Visits Per Month",
                0, 1000, 5
            )

            web = st.number_input(
                "Web Purchases",
                0, 5000, 5
            )

            catalog = st.number_input(
                "Catalog Purchases",
                0, 5000, 3
            )

            store = st.number_input(
                "Store Purchases",
                0, 5000, 7
            )

            campaigns = st.number_input(
                "Campaigns Accepted",
                0, 6, 1
            )


        submit = st.form_submit_button(
            "✨ Predict Customer Segment",
            type="primary",
            use_container_width=True
        )


    # =================================================
    # 10. PREDICTION PROCESS
    # =================================================

    if submit:

        customer = {

            "Age": age,
            "Income": income,
            "Total_Spent": spent,
            "Children": children,
            "Tenure_Days": tenure,
            "Recency": recency,
            "Total_Purchases": purchases,
            "NumDealsPurchases": deals,
            "NumWebVisitsMonth": visits,
            "NumWebPurchases": web,
            "NumCatalogPurchases": catalog,
            "NumStorePurchases": store,
            "Campaigns_Accepted": campaigns

        }

        try:

            input_data = pd.DataFrame(
                [customer]
            ).reindex(columns=FEATURES)

            if input_data.isnull().any().any():

                raise ValueError(
                    "Some required model features are missing."
                )


            # Apply saved preprocessing and model

            scaled_data = model["scaler"].transform(
                input_data
            )

            pca_data = model["pca"].transform(
                scaled_data
            )

            cluster = int(
                model["kmeans"].predict(pca_data)[0]
            )

            predicted_segment = get_segment_name(cluster)


            # SAVE PREDICTION IN SESSION HISTORY

            record = {

                "Predicted At": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                **customer,

                "Predicted Segment": predicted_segment

            }

            st.session_state.prediction_history.append(
                record
            )


            # DISPLAY RESULT

            st.success(
                "Prediction Completed Successfully!"
            )

            st.markdown("### Prediction Result")

            st.markdown(
                f"## 🎯 {predicted_segment}"
            )

            st.markdown("### Suggested Action")

            st.info(
                get_recommendation(predicted_segment)
            )


            # CUSTOMER DETAILS

            with st.expander(
                "View Customer Details Used for Prediction"
            ):

                st.dataframe(
                    pd.DataFrame([customer]),
                    use_container_width=True,
                    hide_index=True
                )


            st.caption(
                "This is a model-based prediction, not a guaranteed "
                "customer outcome. The new customer is not automatically "
                "added to the original dataset."
            )


        except Exception as error:

            st.error(
                "Prediction could not be completed. "
                "Please verify the model and input features."
            )

            st.exception(error)


# =====================================================
# 11. FOOTER
# =====================================================

st.divider()

st.caption(
    "Customer Segmentation Analytics | "
    "Python • Streamlit • Pandas • Plotly • scikit-learn | "
    "PCA + K-Means"
)
