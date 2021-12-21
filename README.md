# SentimentAPI
In this repo you will find a (trained) sentiment annotation model
predicting text to have either positive or negative sentiment with a
certain probability. Furthermore, this repo contains a Flask application
that wraps the model in an API endpoint and adds a simple user
interface.

## Model
The final model is trained on 20% of 4M Reddit posts (see
[https://www.kaggle.com/ehallmar/reddit-comment-score-prediction](https://www.kaggle.com/ehallmar/reddit-comment-score-prediction))
and has a 0.71 ROC AUC score on all blog posts. For more details, see
the different notebooks in `notebooks/`.

The model's final architecture can be described as follows:

1. **Text pre-processing** (a.o. lemmatization, lowercase and removal of
   stopwords)
2. **TF-IDF vecotrization** to transform text into features
3. **Logistic regression** to estimate sentiment probabilities

For more details, please see `modeling/` and `modeling/input` for
(optimal) model inputs.

## App/API
The Flask application can be run by first executing:

```cmd
cd /path/to/this/repo
pip install -r requirements.text
```

Note, preferably you should create a new virtual environment to run this
application.

Once the requirements are installed, you can simply run the app by
executing the following

```cmd
python application.py
```
This will run the app on
[http://localhost:5000/](http://localhost:5000/). The home page contains
a self-explainable user interface, whereas the API can be reached at
[http://localhost:5000/sentiment/**<text>**](http://localhost:5000/sentiment/)
in which **<text>** should be replaced by your text (HTML encoded).

For example, if your text is "Klarna is a great company!" the following
URL should be provided
[http://localhost:5000/sentiment/Klarna%20is%20a%20great%20company%21](http://localhost:5000/sentiment/Klarna%20is%20a%20great%20company%21).
This will return a JSON with probabilities the provided text has either
positive or negative sentiment.

### Deployment on AWS
This application is made publicly available on AWS Elastic Beanstalk at
[http://flask.eba-w7r8vmnk.eu-central-1.elasticbeanstalk.com/](http://flask.eba-w7r8vmnk.eu-central-1.elasticbeanstalk.com/)
