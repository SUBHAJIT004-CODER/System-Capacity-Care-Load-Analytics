
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Healthcare Capacity Analytics",
    layout="wide"
)

st.title("🏥 Healthcare Capacity & Care Load Analytics")

uploaded_file = st.file_uploader(
    "Upload healthcare_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric(
            "Total CBP Custody",
            int(df["Children in CBP custody"].sum())
        )

    with col2:
        st.metric(
            "Total HHS Care",
            int(df["Children in HHS Care"].sum())
        )

    with col3:
        st.metric(
            "Total Transfers",
            int(df["Children transferred out of CBP custody"].sum())
        )

    st.subheader("Trend Analysis")

    fig1 = px.line(
        df,
        y="Children in CBP custody",
        title="CBP Custody Trend"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.line(
        df,
        y="Children in HHS Care",
        title="HHS Care Trend"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Correlation Analysis")

    corr = df.select_dtypes(include="number").corr()

    st.dataframe(corr)

    st.subheader("AI Forecast")

    try:

        model = joblib.load(
            "healthcare_ai_model.pkl"
        )

        day = st.number_input(
            "Day",
            1,
            31,
            15
        )

        month = st.number_input(
            "Month",
            1,
            12,
            7
        )

        year = st.number_input(
            "Year",
            2024,
            2035,
            2026
        )

        cbp = st.number_input(
            "Children in CBP custody",
            value=500
        )

        transfer = st.number_input(
            "Children transferred out of CBP custody",
            value=450
        )

        discharge = st.number_input(
            "Children discharged from HHS Care",
            value=420
        )

        if st.button("Predict"):

            sample = pd.DataFrame({

                "Day":[day],
                "Month":[month],
                "Year":[year],
                "Children in CBP custody":[cbp],
                "Children transferred out of CBP custody":[transfer],
                "Children discharged from HHS Care":[discharge]

            })

            prediction = model.predict(sample)

            st.success(
                f"Predicted HHS Care Load: {prediction[0]:.2f}"
            )

    except:
        st.warning(
            "Upload healthcare_ai_model.pkl in the deployment folder."
        )

else:
    st.info("Upload healthcare_data.csv to begin.")
