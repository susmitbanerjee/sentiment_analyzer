from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

# VADER initialisation
vader = SentimentIntensityAnalyzer()


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/analyze", methods=['POST'])
def analyze():
    if request.method == 'POST':
        review = request.form['review']
        # vader analysis
        scores = vader.polarity_scores(review)

        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= 0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        return render_template('result.html', review=review, sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
