import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")
BASE = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(BASE / "segmentation_model.pkl")

@st.cache_data
def load_customers():
    return pd.read_csv(BASE / "customers_with_segments.csv")

model = load_model()
data = load_customers()
st.markdown("""
<style>
.stApp {background:#f5f7fb}
.block-container {padding-top:1.5rem}
.hero {padding:26px 30px;border-radius:18px;color:white;
background:linear-gradient(110deg,#14213d,#31558a);margin-bottom:20px}
.hero h1 {margin:0}.hero p {color:#e5edff;margin:8px 0 0}
div[data-testid="stMetric"] {background:white;padding:15px;border-radius:14px;border:1px solid #e6eaf2}
</style>
<div class="hero"><h1>Customer Segmentation Analytics</h1>
<p>Explore customer groups, spending patterns and predict a new customer's segment.</p></div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["Overview", "Segment Analysis", "Predict Customer"])
st.sidebar.caption("PCA + K-Means · 3 segments")

if page == "Overview":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers", f"{len(data):,}")
    c2.metric("Segments", data["Segment"].nunique())
    c3.metric("Average spend", f"{data['Total_Spent'].mean():,.0f}")
    c4.metric("Average income", f"{data['Income'].mean():,.0f}")
    left,right = st.columns(2)
    counts = data["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
    with left:
        st.plotly_chart(px.pie(counts,names="Segment",values="Customers",hole=.55,title="Customer distribution"),use_container_width=True)
    with right:
        spend = data.groupby("Segment",as_index=False)["Total_Spent"].mean()
        st.plotly_chart(px.bar(spend,x="Segment",y="Total_Spent",color="Segment",title="Average spend by segment"),use_container_width=True)
    st.subheader("Segment summary")
    summary = data.groupby("Segment").agg(Customers=("Segment","size"),Average_Income=("Income","mean"),
        Average_Spend=("Total_Spent","mean"),Average_Purchases=("Total_Purchases","mean")).round(1).reset_index()
    st.dataframe(summary,use_container_width=True,hide_index=True)
    
    pca_variance = model["pca"].explained_variance_ratio_.sum()

    st.caption(f"PCA variance explained: {pca_variance:.1%}")


elif page == "Segment Analysis":
    st.subheader("Explore a customer segment")
    segment=st.selectbox("Select segment",sorted(data["Segment"].unique()))
    part=data[data["Segment"]==segment]
    a,b,c=st.columns(3)
    a.metric("Customers",f"{len(part):,}")
    b.metric("Average spend",f"{part['Total_Spent'].mean():,.0f}")
    c.metric("Average income",f"{part['Income'].mean():,.0f}")
    product_cols=["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
    prod=part[product_cols].mean().rename_axis("Product category").reset_index(name="Average spend")
    st.plotly_chart(px.bar(prod,x="Product category",y="Average spend",title="Average product spending"),use_container_width=True)
    channel=part[["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"]].mean().rename_axis("Channel").reset_index(name="Average purchases")
    channel["Channel"]=channel["Channel"].replace({"NumWebPurchases":"Web","NumCatalogPurchases":"Catalog","NumStorePurchases":"Store"})
    st.plotly_chart(px.bar(channel,x="Channel",y="Average purchases",title="Purchase channels"),use_container_width=True)
    st.dataframe(part[["ID","Age","Income","Total_Spent","Total_Purchases","Segment"]].head(100),use_container_width=True,hide_index=True)
    st.download_button("Download segment CSV",part.to_csv(index=False).encode(),file_name="customer_segment.csv",mime="text/csv")

else:
    st.subheader("Predict a new customer's segment")
    st.write("Enter the customer's details below.")
    with st.form("prediction"):
        a,b,c=st.columns(3)
        with a:
            age=st.number_input("Age",18,99,35)
            income=st.number_input("Annual income",0,1000000,50000,step=1000)
            spent=st.number_input("Total product spend",0,100000,500,step=50)
            children=st.number_input("Children",0,10,0)
            tenure=st.number_input("Tenure in days",0,10000,365)
        with b:
            recency=st.number_input("Days since last purchase",0,365,30)
            purchases=st.number_input("Total purchases",0,1000,15)
            deals=st.number_input("Deal purchases",0,100,2)
            visits=st.number_input("Web visits per month",0,100,5)
        with c:
            web=st.number_input("Web purchases",0,500,5)
            catalog=st.number_input("Catalog purchases",0,500,3)
            store=st.number_input("Store purchases",0,500,7)
            campaigns=st.number_input("Campaigns accepted",0,6,1)
        submit=st.form_submit_button("Predict segment",type="primary",use_container_width=True)
    if submit:
        customer={"Age":age,"Income":income,"Total_Spent":spent,"Children":children,"Tenure_Days":tenure,
        "Recency":recency,"Total_Purchases":purchases,"NumDealsPurchases":deals,"NumWebVisitsMonth":visits,
        "NumWebPurchases":web,"NumCatalogPurchases":catalog,"NumStorePurchases":store,"Campaigns_Accepted":campaigns}
        X=pd.DataFrame([customer])[model["features"]]
        transformed=model["pca"].transform(model["scaler"].transform(X))
        cluster=int(model["kmeans"].predict(transformed)[0])
        st.success(f"Predicted segment: {model['names'][cluster]}")
        st.info("This is a model-based customer group, not a guaranteed outcome.")

st.divider()
st.caption("Customer Segmentation · Streamlit · scikit-learn · Plotly")
