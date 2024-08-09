from flask import Flask, request, render_template, url_for, redirect
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the saved scaler and model with error handling
scaler = None
model = None
scaler_path = 'savings_scaler.pkl'
model_path = 'savings_model.pkl'

def load_scaler_and_model():
    global scaler, model
    if os.path.exists(scaler_path) and os.path.exists(model_path):
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Scaler and model loaded successfully.")
        except Exception as e:
            print(f"Error loading scaler or model: {str(e)}")
    else:
        print(f"Scaler or model file not found.")

load_scaler_and_model()

def predict_savings_and_time(monthly_payment, total_income, savings_goal):
    if scaler is None or model is None:
        return None, None, None
    
    input_scaled = scaler.transform([[monthly_payment, total_income]])
    predicted_savings = model.predict(input_scaled)[0]
    
    disposable_income = total_income - monthly_payment
    max_savings = disposable_income * 0.5
    predicted_savings = min(predicted_savings, max_savings)
    
    savings_percentage = (predicted_savings / total_income) * 100
    months_to_goal = max(min(int(np.ceil(savings_goal / predicted_savings)), 96), 1) if predicted_savings > 0 else 96
    
    return predicted_savings, savings_percentage, months_to_goal

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        monthly_payment = float(request.form['monthly_payment'])
        total_income = float(request.form['total_income'])
        savings_goal = float(request.form['savings_goal'])
        
        if scaler is None or model is None:
            error = "Scaler or model not loaded. Please check server logs."
        else:
            predicted_savings, savings_percentage, months_to_goal = predict_savings_and_time(monthly_payment, total_income, savings_goal)
            
            if predicted_savings is not None:
                result = {
                    'predicted_savings': predicted_savings,
                    'savings_percentage': savings_percentage,
                    'months_to_goal': months_to_goal,
                    'monthly_payment': monthly_payment,
                    'total_income': total_income,
                    'savings_goal': savings_goal
                }
            else:
                error = "Error occurred during prediction."
    
    return render_template('index.html', result=result, error=error)

@app.route('/saving-locks-dashboard')
def saving_locks_dashboard():
    return render_template('saving-locks-dashboard.html')

@app.route('/expenses-tracker')
def expenses_tracker():
    return render_template('expenses-tracker.html')

@app.route('/setup-form')
def setup_form():
    return render_template('saving-locks-setup-form.html')

@app.route('/process-setup', methods=['POST'])
def process_setup():
    if request.method == 'POST':
        goal = float(request.form['goal'])
        amount = float(request.form['amount'])
        period = float(request.form['period'])
        duration = int(request.form['duration'])
        
        # Here you would typically save this data to a database
        # For now, we'll just pass it as URL parameters
        return redirect(url_for('saving_locks_dashboard', 
                                goal=goal, 
                                amount=amount, 
                                period=period, 
                                duration=duration))

@app.route('/link-to-bank')
def link_to_bank():
    # This route will redirect to the saving locks dashboard
    # In a real application, you would handle bank linking logic here
    return redirect(url_for('saving_locks_dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
