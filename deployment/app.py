import flask
import numpy as np
import pickle


model_lyft = pickle.load(open("model/model_classifier-lyft.pkl", "rb"))
model_uber = pickle.load(open("model/model_classifier-uber.pkl", "rb"))
app = flask.Flask(__name__, template_folder='templates')


@app.route('/')
def main():
    return(flask.render_template('main.html'))


if __name__ == '__main__':
    app.run()


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = dict(flask.request.form)


    features_name = []
    for i in range(1, 7):
        if i == int(features['name']):
            features_name.append(1)
        else:
            features_name.append(0)

    if features['merk'] == 'lyft':
        float_features = [float(features['distance']),
                          float(features['surge_multiplier'])]
    else:
        float_features = [float(features['distance'])]

    float_features += features_name

    final_features = [np.array(float_features)]
    if features['merk'] == 'lyft':
        prediction = model_lyft.predict(final_features)
    else:
        prediction = model_uber.predict(final_features)

    return flask.render_template("main.html",  prediction_text="Prediksi harga taksi anda ${}".format(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True)
