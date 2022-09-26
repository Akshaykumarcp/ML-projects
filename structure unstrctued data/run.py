from io import BytesIO
from flask import Flask, request, make_response, send_file
from stemming.porter2 import stem
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from io import BytesIO

app = Flask(__name__)

def cleanse_text(text):

    if text:
        # remove whitespace
        clean = ' '.join([i for i in text.split()])
        # stemming
        red_text = [stem(word) for word in clean.split()]
        return ' '.join(red_text) # converts from list to string
    else:
        return text
@app.route('/cluster', methods=['post'])
def cluster():
    data = pd.read_csv(request.files['dataset'])

    unstructure = 'text'

    if 'col' in request.args:
        unstructure = request.args.get('col')

    no_of_clusters = 2

    if 'no_of_clusters' in request.args:
        no_of_clusters = request.args.get('no_of_clusters')

    data = data.fillna('NULL')

    data['clean_sum'] = data[unstructure].apply(cleanse_text)

    vectorizer = CountVectorizer(analyzer='word',stop_words='english')

    counts = vectorizer.fit_transform(data['clean_sum'])

    kmeans = KMeans(n_clusters=no_of_clusters)

    data['cluster_num'] = kmeans.fit_predict(counts)

    data = data.drop(['clean_sum'], axis=1)

    output = BytesIO()

    writer = pd.ExcelWriter(output, engine='xlswriter')
    data.to_excel(writer, sheet_name="clusters", encoding='utf-8', index=False)

    return 'this works buddy'

if __name__ == '__main__':
    app.run(host='0.0.0.0')