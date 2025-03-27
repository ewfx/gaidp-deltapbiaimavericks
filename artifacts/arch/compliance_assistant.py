import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from io import BytesIO
import time
import random
from multiprocessing import Process, Queue
from transformers import pipeline

# Function to run the LLM in a separate process
def run_llm_in_process(prompt, queue):
    try:
        llm = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)  # CPU usage
        generated_rules = llm(prompt, max_length=1000)[0]['generated_text']
        queue.put(generated_rules)
    except Exception as e:
        queue.put(f"Error: {str(e)}")

# Function to extract profiling rules using an LLM
def extract_profiling_rules(instructions):
    regulatory_instructions = """
    - Transaction_Amount should always match Reported_Amount, except when the transaction involves cross-currency conversions, in which case a permissible deviation of up to 1% is allowed.
    - Account_Balance should never be negative, except in cases of overdraft accounts explicitly marked with an "OD" flag.
    - Currency should be a valid ISO 4217 currency code, and the transaction must adhere to cross-border transaction limits as per regulatory guidelines.
    - Country should be an accepted jurisdiction based on bank regulations, and cross-border transactions should include mandatory transaction remarks if the amount exceeds $10,000.
    - Transaction_Date should not be in the future, and transactions older than 365 days should trigger a data validation alert.
    - High-risk transactions (amount > $5,000 in high-risk countries) should be flagged, with an automatic compliance check triggered.
    - Round-number transactions (e.g., $1000, $5000) should be analyzed for potential money laundering risks, requiring additional validation steps.
    - A dynamic risk scoring system should be implemented, adjusting scores based on transaction patterns and historical violations.
    """

    default_profiling_rules = {
        'Customer_ID': {'type': 'int', 'rule': lambda x: x > 0, 'null_allowed': False},
        'Account_Balance': {'type': 'float', 'rule': lambda x: x >= 0, 'null_allowed': False},
        'Transaction_Amount': {'type': 'float', 'rule': lambda x: x >= 0, 'null_allowed': False},
        'Reported_Amount': {'type': 'float', 'rule': lambda x: x >= 0, 'null_allowed': False},
        'Currency': {'type': 'str', 'rule': lambda x: x in ['USD', 'EUR', 'GBP', 'JPY'], 'null_allowed': False},
        'Country': {'type': 'str', 'rule': lambda x: x in ['US', 'DE', 'UK', 'JP'], 'null_allowed': False},
        'Transaction_Date': {
            'type': 'datetime',
            'rule': lambda x: pd.to_datetime(x, errors='coerce') <= pd.Timestamp('2025-03-26') and (pd.Timestamp('2025-03-26') - pd.to_datetime(x, errors='coerce')).days <= 365,
            'null_allowed': False
        },
        'Risk_Score': {'type': 'float', 'rule': lambda x: 0 <= x <= 10 if pd.notnull(x) else True, 'null_allowed': True},
        'Transaction_Amount vs Reported_Amount': {
            'cross_field': lambda row: (
                abs(row['Transaction_Amount'] - row['Reported_Amount']) <= st.session_state.deviation * row['Transaction_Amount']
                if row['Currency'] != 'USD' else row['Transaction_Amount'] == row['Reported_Amount']
            )
        },
        'Round_Number_Transaction': {'cross_field': lambda row: row['Transaction_Amount'] % 1000 != 0},
        'High_Risk_Transaction': {'cross_field': lambda row: not (row['Transaction_Amount'] > 5000 and row['Country'] in ['DE'])},
        'Cross_Border_Remarks': {
            'cross_field': lambda row: row['Transaction_Amount'] <= 10000 or (row['Country'] in ['US', 'DE', 'UK', 'JP'] and row['Currency'] in ['USD', 'EUR', 'GBP', 'JPY'])
        }
    }

    prompt = f"Convert the following regulatory instructions into Python validation rules:\n{regulatory_instructions}"
    queue = Queue()
    process = Process(target=run_llm_in_process, args=(prompt, queue))
    process.start()
    process.join(timeout=30)  # Reduced timeout for faster fallback

    if not queue.empty():
        generated_rules = queue.get()
        if generated_rules.startswith("Error:"):
            st.error(f"LLM failed: {generated_rules}. Using default rules.")
            return default_profiling_rules
        st.write("Generated Rules (for demo):", generated_rules)
        return default_profiling_rules  # Still using defaults for simplicity
    else:
        st.warning("LLM timed out. Using default profiling rules.")
        return default_profiling_rules

# Validation function
def validate_dataset(df, rules):
    validation_results = {}
    for column, rule in rules.items():
        if 'cross_field' in rule:
            continue
        if column not in df.columns:
            validation_results[column] = f"Column '{column}' not found."
            continue
        if rule['type'] == 'int' and not pd.api.types.is_integer_dtype(df[column]):
            validation_results[column] = f"Column '{column}' should be integer."
            continue
        elif rule['type'] == 'float' and not pd.api.types.is_float_dtype(df[column]):
            validation_results[column] = f"Column '{column}' should be float."
            continue
        elif rule['type'] == 'datetime':
            df[column] = pd.to_datetime(df[column], errors='coerce')
        if not rule['null_allowed'] and df[column].isnull().any():
            invalid_rows = df[df[column].isnull()].index.tolist()
            validation_results[column] = f"Null values at rows: {invalid_rows}"
            continue
        invalid_rows = df[column][~df[column].apply(rule['rule'])].index.tolist()
        validation_results[column] = f"Invalid values at rows: {invalid_rows}" if invalid_rows else "Valid"

    for column, rule in rules.items():
        if 'cross_field' in rule:
            invalid_rows = df[~df.apply(rule['cross_field'], axis=1)].index.tolist()
            validation_results[column] = f"Cross-field validation failed at rows: {invalid_rows}" if invalid_rows else "Valid"
    return validation_results

# Adaptive risk scoring
def adjust_risk_score(row, validation_results):
    risk_adjustment = 0
    for column, result in validation_results.items():
        if 'rows' in str(result):
            invalid_rows = [int(r) for r in str(result).split('[')[1].split(']')[0].split(',') if r.strip()]
            if row.name in invalid_rows:
                risk_adjustment += 2
    if row['Anomaly'] == -1:
        risk_adjustment += 3
    return row['Risk_Score'] + risk_adjustment if pd.notnull(row['Risk_Score']) else risk_adjustment

# Remediation suggestions
def generate_remediation_suggestions(df, validation_results):
    suggestions = []
    for column, result in validation_results.items():
        if 'rows' in str(result):
            invalid_rows = [int(r) for r in str(result).split('[')[1].split(']')[0].split(',') if r.strip()]
            for row in invalid_rows:
                if column == 'Transaction_Amount vs Reported_Amount':
                    suggestions.append(f"Row {row} (Customer_ID {df.loc[row, 'Customer_ID']}): Adjust Reported_Amount to {df.loc[row, 'Transaction_Amount']}. Reason: {result}")
                elif column in ['Currency', 'Country', 'Transaction_Date']:
                    suggestions.append(f"Row {row} (Customer_ID {df.loc[row, 'Customer_ID']}): Request missing {column} value. Reason: {result}")
                elif column == 'Cross_Border_Remarks':
                    suggestions.append(f"Row {row} (Customer_ID {df.loc[row, 'Customer_ID']}): Add remarks for cross-border transaction > $10,000. Reason: {result}")
                elif column == 'High_Risk_Transaction':
                    suggestions.append(f"Row {row} (Customer_ID {df.loc[row, 'Customer_ID']}): Trigger compliance review for high-risk transaction. Reason: {result}")
                else:
                    suggestions.append(f"Row {row} (Customer_ID {df.loc[row, 'Customer_ID']}): Correct {column} value. Reason: {result}")
    return suggestions

# PDF report generation
def generate_pdf_report(df, validation_results, anomaly_rows, remediation_suggestions):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Data Quality Report")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Validation Results:")
    y -= 20
    c.setFont("Helvetica", 10)
    for column, result in validation_results.items():
        text = f"{column}: {result}"
        c.drawString(50, y, text[:100])  # Truncate to avoid overflow
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Anomaly Detection Results:")
    y -= 20
    c.setFont("Helvetica", 10)
    for _, row in anomaly_rows.iterrows():
        text = f"Customer_ID: {row['Customer_ID']}, Balance: {row['Account_Balance']}, Txn: {row['Transaction_Amount']}"
        c.drawString(50, y, text)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Adaptive Risk Scores:")
    y -= 20
    c.setFont("Helvetica", 10)
    for _, row in df.iterrows():
        text = f"Customer_ID: {row['Customer_ID']}, Risk: {row['Risk_Score']}, Adaptive: {row['Adaptive_Risk_Score']}"
        c.drawString(50, y, text)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Remediation Suggestions:")
    y -= 20
    c.setFont("Helvetica", 10)
    for suggestion in remediation_suggestions:
        c.drawString(50, y, suggestion[:100])  # Truncate for safety
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    buffer.seek(0)
    return buffer

# Simulate new transaction
def simulate_new_transaction(customer_id):
    return {
        'Customer_ID': customer_id,
        'Account_Balance': random.uniform(1000, 100000),
        'Transaction_Amount': random.uniform(100, 5000),
        'Reported_Amount': random.uniform(100, 5000),
        'Currency': random.choice(['USD', 'EUR', 'GBP', 'JPY']),
        'Country': random.choice(['US', 'DE', 'UK', 'JP']),
        'Transaction_Date': pd.Timestamp('2025-03-26'),
        'Risk_Score': random.uniform(0, 10)
    }

# Streamlit app
def main():
    st.title("Interactive Compliance Assistant")

    # Initialize session state
    if 'df' not in st.session_state:
        initial_data = {
            'Customer_ID': [1001, 1002, 1003, 1004],
            'Account_Balance': [15000.0, 32000.0, 5000.0, 70000.0],
            'Transaction_Amount': [500.0, 1200.0, 300.0, 2000.0],
            'Reported_Amount': [500.0, 1200.0, 300.0, 1800.0],
            'Currency': ['USD', 'EUR', None, 'USD'],
            'Country': ['US', 'DE', None, 'US'],
            'Transaction_Date': ['2025-02-25', '2025-02-20', None, '2025-02-28'],
            'Risk_Score': [3.0, 2.0, None, 5.0]
        }
        st.session_state.df = pd.DataFrame(initial_data)
    if 'deviation' not in st.session_state:
        st.session_state.deviation = 0.01

    # File upload
    st.header("Upload Dataset (Optional)")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)

    # Preprocess dataset
    df = st.session_state.df.copy()
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
    df['Country'] = df['Country'].fillna('').str.upper()
    df['Currency'] = df['Currency'].fillna('').str.upper()

    # Rule refinement
    st.header("Conversational Rule Refinement")
    user_input = st.chat_input("E.g., 'Increase deviation to 2%'")
    if user_input:
        if "increase deviation to" in user_input.lower():
            try:
                new_deviation = float(user_input.lower().split("increase deviation to")[1].split("%")[0].strip()) / 100
                st.session_state.deviation = new_deviation
                st.success(f"Deviation updated to {new_deviation * 100}%")
            except ValueError:
                st.error("Specify deviation as a percentage (e.g., 'Increase deviation to 2%')")
        else:
            st.warning("Command not recognized.")

    # Extract rules
    profiling_rules = extract_profiling_rules("")

    # Real-time monitoring
    st.header("Real-Time Monitoring")
    monitor = st.checkbox("Enable real-time monitoring", value=False)
    if monitor:
        placeholder = st.empty()
        customer_id_counter = df['Customer_ID'].max() + 1 if not df.empty else 1001
        while st.session_state.get('monitor', True):
            new_transaction = simulate_new_transaction(customer_id_counter)
            df = pd.concat([df, pd.DataFrame([new_transaction])], ignore_index=True)
            st.session_state.df = df
            customer_id_counter += 1

            with placeholder.container():
                display_results(df, profiling_rules)
            time.sleep(5)
    else:
        display_results(df, profiling_rules)

def display_results(df, profiling_rules):
    st.write("Dataset Preview:")
    st.dataframe(df)

    st.header("Adjust Profiling Rules")
    st.session_state.deviation = st.slider("Deviation for Transaction_Amount vs Reported_Amount (%)", 0.0, 10.0, st.session_state.deviation * 100) / 100

    st.header("Validation Results")
    results = validate_dataset(df, profiling_rules)
    st.write(results)

    st.header("Anomaly Detection")
    features = ['Account_Balance', 'Transaction_Amount', 'Reported_Amount']
    X = df[features].fillna(0)  # Handle NaN for anomaly detection
    iso_forest = IsolationForest(contamination=0.25, random_state=42)
    df['Anomaly'] = iso_forest.fit_predict(X)
    anomaly_rows = df[df['Anomaly'] == -1]
    st.write("Anomalies:", anomaly_rows)

    st.header("Adaptive Risk Scores")
    df['Adaptive_Risk_Score'] = df['Risk_Score'].fillna(0)
    df['Adaptive_Risk_Score'] = df.apply(lambda row: adjust_risk_score(row, results), axis=1)
    st.write(df[['Customer_ID', 'Risk_Score', 'Adaptive_Risk_Score', 'Anomaly']])

    st.header("Remediation Suggestions")
    suggestions = generate_remediation_suggestions(df, results)
    for suggestion in suggestions:
        st.write(suggestion)

    st.header("Download Report")
    pdf_buffer = generate_pdf_report(df, results, anomaly_rows, suggestions)
    st.download_button(
        label="Download PDF",
        data=pdf_buffer,
        file_name="data_quality_report.pdf",
        mime="application/pdf"
    )

if __name__ == "__main__":
    main()