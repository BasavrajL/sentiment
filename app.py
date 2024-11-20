from flask import Flask, request, jsonify, render_template
import pickle
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)


logging.basicConfig(level=logging.DEBUG)


with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        simpletext = data['simpletext']
        logging.debug(f"Received text: {simpletext}")
        
        # Get VADER sentiment
        vader_sentiment = get_vader_sentiment(simpletext)
        logging.debug(f"VADER Sentiment: {vader_sentiment}")
        
        # Get ML model sentiment
        model_prediction = model.predict([simpletext])[0]
        logging.debug(f"Model Prediction: {model_prediction}")
        
        return jsonify({
            'vader_sentiment': vader_sentiment,
            'model_sentiment': model_prediction
        })
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
