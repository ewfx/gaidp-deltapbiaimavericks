# 🚀 Interactive Compliance Assistant

## 🎯 Introduction
The Interactive Compliance Assistant is a financial transaction validation system developed for the Wells Fargo Hackathon 2025. It addresses the challenge of ensuring compliance with regulatory guidelines by automating transaction validation, detecting anomalies, and providing remediation suggestions. The project aims to streamline compliance workflows, reduce manual errors, and enable real-time monitoring for financial institutions.
 
🖼️ Screenshots:

![Screenshot 1](link-to-image)

## 💡 Inspiration
The inspiration for this project came from the need to improve financial compliance processes. Manual validation of transactions is time-consuming and prone to errors, especially with strict regulatory requirements like matching transaction amounts, validating currencies, and flagging high-risk transactions. We aimed to create an automated, interactive solution that empowers compliance officers to monitor transactions efficiently and take action on potential issues.

## ⚙️ What It Does
The Interactive Compliance Assistant offers the following key features:

Transaction Validation: Validates financial transactions against regulatory rules (e.g., Transaction_Amount vs. Reported_Amount within 1% deviation, valid ISO 4217 currency codes).
Anomaly Detection: Uses IsolationForest to identify unusual transactions (e.g., discrepancies in reported amounts).
Adaptive Risk Scoring: Dynamically adjusts risk scores based on validation failures and anomalies.
Remediation Suggestions: Provides actionable suggestions for non-compliant transactions (e.g., adjust amounts, flag for review).
Real-Time Monitoring: Simulates new transactions every 5 seconds in the Streamlit app for continuous monitoring.
Interactive Interface: A Streamlit app allows users to upload datasets, adjust rules (e.g., deviation threshold), and download PDF reports.
LLM Integration: Attempts to generate validation rules using bigcode/starcoder, with a manual fallback.

## 🛠️ How We Built It
We developed the project in two main components:

Jupyter Notebook (transaction_validator.ipynb): Handles data processing, validation, anomaly detection, risk scoring, remediation, and LLM-based rule generation.
Streamlit App (compliance_assistant.py): Provides an interactive interface for compliance monitoring, real-time updates, and reporting.
We used a sample dataset (sample_data.csv) to test the system, simulating financial transactions with issues like missing data and discrepancies.

## 🚧 Challenges We Faced
LLM Integration: The initial LLM (gpt2) failed to generate usable rules, so we switched to bigcode/starcoder. However, we faced repeated download timeouts for the model files, forcing us to rely on manual rules.
Hugging Face Authentication: Accessing bigcode/starcoder required Hugging Face authentication, which added an extra step.
Dataset Limitations: The sample dataset is small (4 rows), limiting the robustness of anomaly detection and testing.

## 🏃 How to Run
1. Clone the repository  
   ```sh
   https://github.com/ewfx/gaidp-deltapbiaimavericks/tree/main
   ```
2. Create a virtual environment and install dependencies:  
   ```sh
   python -m venv myenv
   myenv\Scripts\activate  # On Windows
   # source myenv/bin/activate  # On macOS/Linux
   pip install -r requirements.txt

3. Run the Jupyter notebook:
   ```sh
   jupyter notebook transaction_validator.ipynby
   ```
4.Run the Streamlit app:
   ```sh
   streamlit run compliance_assistant.py
   ```

## 🏗️ Tech Stack
-🔹 Python: Core programming language.
-🔹 Pandas & NumPy: Data manipulation and preprocessing.
-🔹 Scikit-learn: IsolationForest for anomaly detection.
-🔹 Transformers (Hugging Face): bigcode/starcoder for LLM-based rule generation.
-🔹 Streamlit: Interactive web app for compliance monitoring.
-🔹 ReportLab: PDF report generation.
-🔹 Other: Jupyter Notebook for development and testing.

## 👥 Team
- **Avinash Nalla**
- **Krishna Kodati**

