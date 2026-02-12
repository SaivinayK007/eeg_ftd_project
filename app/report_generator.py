from fpdf import FPDF
from datetime import datetime

def generate_report(subject, age, mmse, prediction, risk, confidence, features):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "EEG Dementia Risk Assessment Report", ln=True)
    pdf.ln(5)

    pdf.cell(0, 8, f"Subject: {subject}", ln=True)
    pdf.cell(0, 8, f"Age: {age}", ln=True)
    pdf.cell(0, 8, f"MMSE: {mmse}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 8, f"Risk: {risk}", ln=True)
    pdf.cell(0, 8, f"Prediction: {prediction}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", ln=True)

    pdf.ln(5)
    pdf.cell(0, 8, "EEG Mean Features:", ln=True)
    names = ["Delta", "Theta", "Alpha", "Beta", "Entropy"]
    for n, v in zip(names, features):
        pdf.cell(0, 8, f"{n}: {v:.4f}", ln=True)

    pdf.ln(5)
    pdf.multi_cell(0, 8, "Disclaimer: Research decision-support only.")
    pdf.output("EEG_Clinical_Report.pdf")
