from flask import Flask, render_template, request
import paho.mqtt.client as mqtt

app = Flask(__name__)

def on_connect(client, userdata, flags, rc):
    print("Connected to server (i.e., broker) with result code " + str(rc))
    # subscribe to topics of interest here

# Default message callback. Please use custom callbacks.
def on_message(client, userdata, msg):
    print("on_message: " + msg.topic + " " + str(msg.payload, "utf-8"))

@app.route("/")
def home():
    return render_template("index.html")

# GET request endpoint 2
@app.route("/send", methods=["POST", "GET"])
def eth_home():
    msg = ''
    if 'Message' in request.args.keys():
        msg = request.args['Message']
    client.publish("ITP388/test", msg)
    return render_template("index.html", returnMsg=msg)


if __name__ == "__main__":
    # this section is covered in publisher_and_subscriber_example.py
    client = mqtt.Client()
    client.on_message = on_message
    client.on_connect = on_connect
    client.connect(host="broker.hivemq.com", port=1883, keepalive=60)
    client.loop_start()
    app.run(debug=True)