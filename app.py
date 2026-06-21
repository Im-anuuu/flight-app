from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

# Load model and transformer
model       = pickle.load(open('flight_model.pkl', 'rb'))
transformer = pickle.load(open('flight_transformer.pkl', 'rb'))

# Check if model is a Pipeline (includes transformer inside)
from sklearn.pipeline import Pipeline
IS_PIPELINE = isinstance(model, Pipeline)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        airline        = request.form['airline']
        source_city    = request.form['source_city']
        destination    = request.form['destination_city']
        departure_time = request.form['departure_time']
        stops          = request.form['stops']
        arrival_time   = request.form['arrival_time']
        flight_class   = request.form['class']
        duration       = float(request.form['duration'])
        days_left      = int(request.form['days_left'])

        def make_df(d_left):
            return pd.DataFrame([{
                'airline':          airline,
                'source_city':      source_city,
                'destination_city': destination,
                'departure_time':   departure_time,
                'stops':            stops,
                'arrival_time':     arrival_time,
                'class':            flight_class,
                'duration':         duration,
                'days_left':        d_left,
            }])

        def predict_price(df):
            # if model is Pipeline, it handles transformation internally
            if IS_PIPELINE:
                return model.predict(df)[0]
            else:
                return model.predict(transformer.transform(df))[0]

        # ── Single prediction ─────────────────────────────────────────────────
        prediction = predict_price(make_df(days_left))
        result     = f'₹{int(prediction):,}'

        # ── Price trend: loop days_left 1 → 49 ───────────────────────────────
        trend_days   = list(range(1, 50))
        trend_prices = [int(predict_price(make_df(d))) for d in trend_days]

        chosen_day   = days_left
        chosen_price = int(prediction)

        chart_data = json.dumps({
            'days':         trend_days,
            'prices':       trend_prices,
            'chosen_day':   chosen_day,
            'chosen_price': chosen_price,
        })

    except Exception as e:
        result     = f'Prediction error: {e}'
        chart_data = None
        chosen_day = chosen_price = None

    return render_template(
        'index.html',
        result=result,
        chart_data=chart_data,
        chosen_day=chosen_day,
        chosen_price=chosen_price,
    )


#@app.route('/how-it-works')
#def how_it_works():
#    return render_template('how_it_works.html')


#@app.route('/about')
#def about():
#    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
