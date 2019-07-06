import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.plotly as py
import plotly.graph_objs as go
from joblib import load
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df_clean', engine)

#cols = ['related', 'request', 'offer',
#       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
#       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
#       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
#       'infrastructure_related', 'transport', 'buildings', 'electricity',
#       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
#       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
#       'other_weather', 'direct_report']

#df[cols] = df[cols].apply(pd.to_numeric)

# load model
model = load("../models/classifier.pkl")


def create_barchart(df):
    """Create a plotly figure of a messages per category barplot
    Parameters
    ----------
    df : pandas.Dataframe
        Preprocessed dataset stored in database
    Returns
    -------
    fig1:
        Plotly bar chart
    """

    # Count class occurences
    class_data = df.iloc[:, 4:].sum().to_frame().reset_index()
    class_data.columns = ['class', 'total']
    class_data = class_data.sort_values(by='total', ascending=False)

    data = [go.Bar(
            x=class_data['class'],
            y=class_data['total']
    )]

    layout = go.Layout(
        title='Distribution of Message Classes',
        xaxis=dict(
            title='Labels',
            tickangle=45
        ),
        yaxis=dict(
            title='Occurences',
            tickfont=dict(
                color='DarkGreen')
        )
    )

    fig1 = go.Figure(data=data, layout=layout)

    return fig1


def create_histogram(df):
    """Create a plotly histogram figure
    Parameters
    ----------
    df : pandas.Dataframe
        The dataset
    Returns
    -------
    fig2:
        The plotly histogram
    """

    hist_data = df.iloc[:, 4:].sum(axis=1)
    data = [go.Histogram(x=hist_data)]

    layout = go.Layout(
        title='Histogram of Classes per Message',
        xaxis=dict(
            title='# of Labels per Message',
            tickangle=45
        ),
        yaxis=dict(
            title='Occurences',
            tickfont=dict(
                color='DarkGreen')
        )
    )
    fig2 = go.Figure(data=data, layout=layout)

    return fig2

# get figures and top categories
fig1 = create_barchart(df)
fig2 = create_histogram(df)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
     
    # create visuals
    graphs = [fig1, fig2]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    #print(classification_labels)
    classification_results = dict(zip(df.columns[4:], classification_labels))
    #print(classification_results)
    for category, classification in classification_results.items():
        if classification == "1":
            print(category)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
