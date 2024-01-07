from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
from tensorflow import keras
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

model = keras.models.load_model("model_kelembaban_tanah.h5")

def classify_moisture(moisture):
    input_data = np.array([[moisture]])
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction)
    class_labels = ['basah', 'normal', 'kering']
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

def display_prediction(moisture, classification_result):
    print(f"Received moisture data from MQTT: {moisture}, Classification: {classification_result}")
    socketio.emit('moisture_update', {'moisture': moisture, 'classification': classification_result})

def on_message(client, userdata, msg):
    try:
        moisture = float(msg.payload.decode('utf-8'))
        classification_result = classify_moisture(moisture)
        display_prediction(moisture, classification_result)
    
    except ValueError:
        print("Received invalid moisture data. Could not convert to float.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_broker_address = "broker.mqtt-dashboard.com"
mqtt_topic = "mantap1"
mqtt_client.connect(mqtt_broker_address, 1883, 60)
mqtt_client.subscribe(mqtt_topic)
mqtt_client.loop_start()

@app.route('/predict_moisture', methods=['POST'])
def predict_moisture():
    try:
        data = request.json
        moisture = float(data.get('moisture'))
        classification_result = classify_moisture(moisture)
        display_prediction(moisture, classification_result)
        return jsonify({'moisture': moisture, 'classification': classification_result})
    except ValueError:
        return jsonify({'error': 'Invalid moisture data. Could not convert to float.'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500

@app.route('/')
def index():
    return render_template('kelembaban_tanah.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

socketio.run(app)
