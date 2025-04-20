import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify, send_file
from waitress import serve
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import textwrap
import requests
from flask_cors import CORS


checkpoint_path = "./deepseek_finetuned_final2"
base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)

# Set model to evaluation mode
model.eval()

app = Flask(__name__)
CORS(app)


def generate_recommendations(financial_data):
    """
    Generate financial recommendations based on key financial metrics and provide some comparison with competitors.
    """
    formatted_financial_data = "\n".join([f"Year {entry['year']}:\n{entry}\n" for entry in financial_data])
    
    latest_entry = max(financial_data, key=lambda x: x["year"])
    latest_data = latest_entry

    
    input_text = f"""
    Analyze the company's multi-year financial data and provide insights , and compare it with competitors:

    Financial Data:
    {formatted_financial_data}
    
    Compare the financial data with competitors.

    Provide actionable recommendations categorized under:
    - Revenue Growth Strategies
    - Cost Reduction & Efficiency
    - Investment & Expansion Plans
    - Risk Management & Compliance
    """

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )

    recommendations_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    recommendations = {}
    current_section = None
    for line in recommendations_text.split("\n"):
        line = line.strip()
        if line.endswith(":"):
            current_section = line[:-1]
            recommendations[current_section] = []
        elif current_section and line:
            recommendations[current_section].append(line)

    return recommendations

def create_pdf(recommendations):
    """Generate a structured PDF using ReportLab."""
    pdf_filename = "mprefinal.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "Financial Recommendations Report")
    
    y_position = height - 80
    line_height = 15
    max_lines_per_page = int((height - 100) / line_height)
    line_count = 0

    left_margin = 100
    right_margin = 20
    text_width = width - left_margin - right_margin

    for section, recs in recommendations.items():
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left_margin, y_position, section)
        y_position -= line_height
        line_count += 1
        
        c.setFont("Helvetica", 10)
        for rec in recs:
            wrapped_text = textwrap.fill(rec, width=text_width // 6)
            for line in wrapped_text.splitlines():
                if y_position - line_height < 50:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y_position = height - 50
                    line_count = 0

                c.drawString(left_margin, y_position, f"- {line}")
                y_position -= line_height
                line_count += 1
                
                if line_count >= max_lines_per_page:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y_position = height - 50
                    line_count = 0

    c.save()
    return pdf_filename


@app.route('/generate_recommendations/<int:user_id>', methods=['POST'])
def generate_recommendations_endpoint(user_id):
    app.logger.info(f'Sending POST request to /generate/{user_id}')
    response = requests.post(f'http://127.0.0.1:5001/generate/{user_id}')
    if response.status_code != 200:
        app.logger.error(f"Failed to fetch financial data: {response.status_code}")

    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch financial data"}), response.status_code

    data = response.json()

    if 'financial_data' not in data:
        return jsonify({"error": "Missing 'financial_data' in response"}), 400

    financial_data = data['financial_data']
    recommendations = generate_recommendations(financial_data)
    pdf_filename = create_pdf(recommendations)
    with open(pdf_filename, 'rb') as f:
        files = {'file': (pdf_filename, f, 'application/pdf')}
        try:
            forward_response = requests.post(f'http://127.0.0.1:5001/upload_pdf/{user_id}',files=files)
            if forward_response.status_code != 200:
                app.logger.warning(f"Forwarding PDF failed: {forward_response.status_code}")
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Error forwarding PDF: {str(e)}")

    return send_file(pdf_filename, as_attachment=True)


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)