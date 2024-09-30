#This is the code used from the implementation of the Main program of AgroTips. 


import ibm_boto3
from ibm_botocore.client import Config
import sqlite3
import logging
import datetime
from datetime import datetime
import pytz
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import paho.mqtt.client as mqtt
import random
import json
import time
import requests
import os
import smtplib
import json
from email.message import EmailMessage
import ssl
import base64
from threading import Event
from typing import Dict, Any

email_sender = 'your email (the code that is used is created for outlook emails, but you can change it by selecting a different service in the follwoing line: with smtplib.SMTP('smtp.office365.com', 587) as smtp:)'

email_password = 'your credential key'

email_receiver = 'The user's email'

subject = 'Check Out Your New AgroTips Report!!'




#The information we need in order to connect to the IBM cloud
api_key = 'IBM Cloud Object Storage API key'
service_instance_id = 'The service instrunce you want to use'
auth_endpoint = 'What kind of authorization you want to use'
service_endpoint = 'The storage domain you chose'

#The folders we want to create inside the cloud in order to have it more organised
image_bucket_name = 'agrotipsimages' #if not in existance, we will create a folder named "agrotipsimages". There, we will store images we collected from our camera.
sensor_data_bucket_name = 'agrotipsensordata' #if not in existance, we will create a folder named "agrotipsdata". There, we will store the data we collected from our sensors.

# Create COS client
cos = ibm_boto3.client(
    "s3",
    ibm_api_key_id=api_key,
    ibm_service_instance_id=service_instance_id,
    ibm_auth_endpoint=auth_endpoint,
    config=Config(signature_version="oauth"),
    endpoint_url=service_endpoint
)

#Now we will check if the bucket we want to create is already uploaded in the cloud or if we need to create it. This way, only teh credentials are needed to setup the IBM Cloud Object Storage

#Bucket for the images
existing_buckets = cos.list_buckets()

# Check if the desired bucket exists
bucket_exists = any(bucket['Name'] == image_bucket_name for bucket in existing_buckets['Buckets'])

if not bucket_exists:
    cos.create_bucket(Bucket=image_bucket_name)
    print("Bucket created successfully")
else:
    print("Bucket already exists")

#Bucket for sensor data
existing_buckets = cos.list_buckets()

# Check if the desired bucket exists
bucket_exists = any(bucket['Name'] == sensor_data_bucket_name for bucket in existing_buckets['Buckets'])

if not bucket_exists:
    cos.create_bucket(Bucket=sensor_data_bucket_name)
    print("Bucket created successfully")
else:
    print("Bucket already exists")


#Here we process the image we created, in order to make it a suitable input for the image recognition model
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format

    # Resize the image to 512x512 using LANCZOS resampling
    image = ImageOps.fit(image, (512, 512), Image.Resampling.LANCZOS)

    # Convert the image to a NumPy array
    image_array = np.asarray(image)

    # Ensure the array is of type uint8 (since no normalization)
    image_array = image_array.astype(np.uint8)

    #Converting image into a tensor, in order to change its contrast and brightness using the tensorflow
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

    #Changing the contrast of the image to 0.55, in order to reveal more information
    image_tensor = tf.image.adjust_contrast(image_tensor, contrast_factor=0.55)

    #Changing the brightness of the image in order extract more information
    image_tensor = tf.image.adjust_brightness(image_tensor, brightness_factor=0.55)

    # Add batch dimension (model expects a batch, even if it's a batch of 1)
    data = np.expand_dims(image_tensor, axis=0)

    return data

#We add the f1score metric we created for the model. It needs this metric n order to run properly.
class F1Score(metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=1)
        y_pred = tf.cast(y_pred, tf.int32)

        # Calculate True Positives, False Positives, False Negatives
        tp = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred) & tf.equal(y_true, 1), self.dtype))
        fp = tf.reduce_sum(tf.cast(tf.not_equal(y_true, y_pred) & tf.equal(y_pred, 1), self.dtype))
        fn = tf.reduce_sum(tf.cast(tf.not_equal(y_true, y_pred) & tf.equal(y_true, 1), self.dtype))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_score

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)







#We Will use the try method and in order to have the programm running repeatedly. Next, we will use time.sleep to make it run every 5 hours (We don't want the system to spam messages 5 five hours is a good time window)

try:
  while True:






    # Define MQTT broker details

    broker = "test.mosquitto.org"
    port = 1883
    topic = "Example-AgroTips132/upload"
    image_path = "/.../your_image_path.jpg"  # Path to the image to be uploaded
    download_path = "downloaded_image.jpg" #Path to the image that will be downloaded

    # Creation of random sensor data
    sensor_data: Dict[str, float] = {
        "Temperature": round(random.uniform(15.0, 30.0), 2),
        "Humidity": round(random.uniform(30.0, 90.0), 2),
        "pH": round(random.uniform(5.5, 7.5), 2),
        "Electrical conductivity": round(random.uniform(0.1, 2.0), 2),
        "Soil Moisture": round(random.uniform(10.0, 50.0), 2),
        "Lux": round(random.uniform(1000.0,80000.0),2)
    }

    # Shared data structure and event
    received_data: Dict[str, Any] = {}
    sub_ready_event = Event()
    data_ready_event = Event()
    publish_ready_event = Event()

    def on_publish(client: mqtt.Client, userdata: Any, mid: int) -> None:
        print("Image and sensor data uploaded successfully.")

    def on_connect_publish(client: mqtt.Client, userdata: Any, flags: Dict[str, int], rc: int) -> None:
        if rc == 0:
            print("Connected to broker for publishing")
            try:
                with open(image_path, 'rb') as file:
                    file_content = file.read()
                encoded_image = base64.b64encode(file_content).decode('utf-8')
                payload = {
                    'image': encoded_image,
                    'info': sensor_data
                }
                payload_str = json.dumps(payload)
                result = client.publish(topic, payload_str)
                print(f"Publish result: {result}")
            except Exception as e:
                print(f"Failed to publish message: {e}")
        else:
            print("Connection failed with code", rc)
        publish_ready_event.set()

    def on_message(client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        print("Message received")
        print(f"Topic: {msg.topic}\nMessage: {msg.payload}")
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            print("Payload parsed successfully.")
            image_data = base64.b64decode(payload['image'])
            with open(download_path, 'wb') as file:
                file.write(image_data)
            print("Image downloaded and saved successfully.")

            info = payload['info']
            received_data['Temperature'] = info['Temperature']
            received_data['Humidity'] = info['Humidity']
            received_data['pH'] = info['pH']
            received_data['Electrical Conductivity'] = info['Electrical conductivity']
            received_data['Soil Moisture'] = info['Soil Moisture']
            received_data['Lux'] = info['Lux']
            data_ready_event.set()
        except Exception as e:
            print(f"Failed to process message: {e}")

    def on_connect_subscribe(client: mqtt.Client, userdata: Any, flags: Dict[str, int], rc: int) -> None:
        if rc == 0:
            print("Connected to broker for subscribing")
            client.subscribe(topic)
        else:
            print("Connection failed with code", rc)
        sub_ready_event.set()

    def on_log(client: mqtt.Client, userdata: Any, level: int, buf: str) -> None:
        print(f"Log: {buf}")

    def file_timestamp(path: str) -> float:
        return os.path.getmtime(path) if os.path.exists(path) else 0

    # Save initial timestamp of the file if it exists
    initial_timestamp = file_timestamp(download_path)

    #Subscribing
    client_subscribe = mqtt.Client()
    client_subscribe.on_message = on_message
    client_subscribe.on_connect = on_connect_subscribe
    client_subscribe.on_log = on_log

    client_subscribe.connect(broker, port)
    client_subscribe.loop_start()
    sub_ready_event.wait(20)

    #Publishing
    client_publish = mqtt.Client()
    client_publish.on_publish = on_publish
    client_publish.on_connect = on_connect_publish
    publish_ready_event.wait(20)

    client_publish.connect(broker, port)
    client_publish.loop_start()
    time.sleep(5)
    client_publish.loop_stop()
    client_publish.disconnect()



    print("Waiting for image and data...")
    data_ready_event.wait(20)

    #We disconnect from the subscribing loop, after the code for publishing is completed, otherwise we might not receive an image. The even code, is used in order to be sure that no errors will occur, you can remove it, but there is a small chance you will experience errors.
    client_subscribe.loop_stop()
    client_subscribe.disconnect()

    # Check if the file was updated
    if file_timestamp(download_path) > initial_timestamp:
        print("Received data:", received_data)
    else:
        print("Image file was not updated.")


    #We change the image path to the path of the image that was downloaded from MQTT
    image_path = ".../downloaded_image.jpg"





    #Now we will upload the image to the bucket with name agrotipsimages

    #Read the Image File
    with open(image_path, 'rb') as file:
        image_data = file.read()

    #Upload the Image
    #I will name the image after the time and date that we have in greece to know when the shot was taken by just reading the name.

    # Get the current time in UTC
    now_utc = datetime.now(pytz.utc)

    # Convert it to Greece's time zone 
    greece_time = now_utc.astimezone(pytz.timezone('Europe/Athens'))

    # Format the time to a string.  Example of what tme_str contains '2024-01-14_15-30-00'
    time_str = greece_time.strftime('%Y-%m-%d_%H-%M-%S')

    #Posting the image
    response = cos.put_object(
        Bucket=image_bucket_name,
        Key=f'Image {time_str}.jpg',
        Body=image_data,
        ContentType='image/jpeg'
    )

    # Check if the upload was successful
    if response and response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print("Upload successful")
    else:
        print("Upload failed")






    #Sql database creation.

    #Connect to the data base we want to create
    connection = sqlite3.connect("sensor_data.db")

    cursor = connection.cursor()

    #Create the table that we will store our sensor data
    cursor.execute("CREATE TABLE IF NOT EXISTS sensor_data (Date_and_time_collected TEXT, ec FLOAT, pH FLOAT, Humidity FLOAT, Temperature FLOAT, Soil_moisture FLOAT, Lux FLOAT)")

    #Create the datalist that we will post
    datalist = [
        (f'{time_str}', received_data['Temperature'], received_data["Humidity"], received_data["pH"], received_data["Electrical Conductivity"], received_data["Soil Moisture"], received_data["Lux"])
    ]
    #Insert the data to the table
    cursor.executemany("INSERT INTO sensor_data (Date_and_time_collected, Temperature, Humidity, pH, ec, Soil_moisture, Lux) VALUES (?, ?, ?, ?, ?, ?, ?)", datalist)

    #Close connection to the database now that we are done
    connection.close()

    # Path to your .db file (the database is the db file)
    db_file_path = '/content/sensor_data.db'

    # Read the .db file in order to create the body
    with open(db_file_path, 'rb') as file:
      db_data = file.read()

    #Posting the database
    response = cos.put_object(
        Bucket=sensor_data_bucket_name,
        Key=f'Sensor Data {time_str}.db',
        Body=db_data,
        ContentType='db'
    )

    # Check if the upload was successful
    if response and response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print("Upload successful")
    else:
        print("Upload failed")

    #///////////////////////////////////////////////////////////////////////////////////////////////////////////////


    # Now it is time to check the image we have taken for any diseases.


    #Our class names.
    class_names= ['Healthy', 'Pseudomonas-Xanthomonas-Septoria', 'Alternaria', 'Cladosporium Leaf Mold', 'Downy Mildew', 'Ash Rot', 'Powdery Mildew']


    #Here we run the model we created
    model = tf.keras.models.load_model('model.keras', custom_objects={'F1Score': F1Score})

    #we print the model summary
    model.summary()


    # The model compilation - make sure ou use the same metrics that you used when creating the model. 
    model.compile(
        optimizer='adam',  
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', F1Score()]
    )

    # Print the model summary to verify.
    model.summary()

    #We call the image processing function we saw earlier
    data = preprocess_image(image_path)

    # Make predictions
    prediction = model.predict(data)
    predicted_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_index]
    confidence_score = prediction[0][predicted_index]

    # Print the prediction and confidence score
    print(f"Class: {predicted_class_name}, Confidence Score: {confidence_score:.2f}")





    #At this point, we need to create custom messages based on the date we received (image, and sensor data) in order to send an email to the user.

    #This will be the part of the email that we create the message of what our system understood based on the image it received
    if not predicted_class_name == "Healthy":
      image_message = """<p><span style="color: green;"><strong>After we looked at your latest image:</strong></span><br>

<br>Our system has indicated that your plants might have """ + predicted_class_name + """. Check the image. If there are is nothing interfering with it, we advice you to contact an agronomist.</p>"""

    else:
      image_message = """"<span style="color: green;"><strong>After we looked at your image:</strong></span><br>

Our system has indicated that your plants look healthy."""




    #We will also send the data that "system results" are based of - sensor data and the image.
    #Here we convert sensor data to str in order add them in our email
    Electrical_Conductivity = str(received_data['Electrical Conductivity'])
    pH = str(received_data['pH'])
    Humidity = str(received_data['Humidity'])
    Temperature = str(received_data['Temperature'])
    Soil_Moisture = str(received_data['Soil Moisture'])
    Lux = str(received_data['Lux'])

    #The sensor data as they they will apeal in the mail.
    sensor_message = """<p><span style="color: green;"><strong>Here are the results from our sensors:</strong></span><br>
<ul>
<li>Electrical Conductivity: """ + Electrical_Conductivity + """</li><br>
<li>pH: """ + pH + """</li><br>
<li>Humidity: """ + Humidity + """</li><br>
<li>Temperature: """ + Temperature + """</li><br>
<li>Soil Moisture: """ + Soil_Moisture + """</li><br>
<li>Lux: """ + Lux + """</li></ul>
<br>
"""

    #Add the tips from the sensor data in sensor_message.
    if received_data['Electrical Conductivity'] < 2:
        sensor_message += """<span style="color: #f44336;">Electrical conductivity is below 2:</span> The plants are undernourished. Increase fertilizer dosage.<br>

"""
    elif received_data['Electrical Conductivity'] > 4:
        sensor_message +=  """<span style="color: #f44336;">Electrical conductivity exceeds 4:</span> There are signs of overfeeding. Apply more water and reduce fertilizer input.<br>

"""

    if received_data['pH'] < 5.8:
        sensor_message += """<span style="color: #f44336;">pH below 5.8 detected:</span> Risk of toxicity. Incorporate more calcium into the fertilizer mix.<br>

"""
    elif received_data['pH'] > 7:
        sensor_message += """<span style="color: #f44336;">pH exceeds 7:</span> Alkalinity is too high. We recommend adding nitric acid to the watering solution to lower pH.<br>

"""

    if received_data['Humidity'] < 40:
        sensor_message += """<span style="color: #f44336;">Atmospheric humidity below 40%:</span> The enviroment is too dry, potentially impairing nutrient uptake. Increase humidity to mitigate.<br>

"""
    elif received_data['Humidity'] > 70:
        sensor_message += """<span style="color: #f44336;">Atmospheric humidity above 70%:</span> Excessive moisture detected, elevating risk of fungal and bacterial diseases. Improve ventilation, possibly by opening windows, to lower humidity.<br>

"""

    if received_data['Temperature'] < 5:
        sensor_message += """<span style="color: #f44336;">Warning!!! Temperature below 5°C:</span> Immediate risk of freezing. Activate heating systems without delay.<br>

"""
    elif received_data['Temperature'] < 12:
        sensor_message += """<span style="color: #f44336;">Temperature below 12°C:</span> The enviroment is suboptimal for nutrient uptake. Advise activating heating to optimize conditions.<br>

"""
    elif received_data['Temperature'] > 28:
        sensor_message += """<span style="color: #f44336;">Temperature exceeds 28°C:</span> Heat stress detected. Engage cooling systems to prevent adverse effects on plant health.<br>

"""

    if received_data['Soil Moisture'] < 20:
        sensor_message += """<span style="color: #f44336;">Critical!!! Soil moisture below 20%:</span> Immediate watering is required to prevent crop loss.<br>

"""
    elif received_data['Soil Moisture'] < 25:
        sensor_message += """<span style="color: #f44336;">Soil moisture below 25%:</span> Insufficient for healthy growth. Watering recommended.<br>

 """
    elif received_data['Soil Moisture'] > 60:
        sensor_message += """<span style="color: #f44336;">Alert!!! Soil moisture above 60%:</span> Halt watering to avoid root diseases and crop damage until levels drop below 55%.<br>

"""
    elif received_data['Soil Moisture'] > 55:
        sensor_message += """<span style="color: #f44336;">Soil moisture exceeds 55%:</span> Reduce frequency or volume of watering to prevent over-saturation.<br>

"""

    month = datetime.now().month
    hour = datetime.now().hour

    # Determine the season based on the month
    is_summer = month in [6, 7, 8]
    is_winter = month in [12, 1, 2]
    is_spring = month in [3, 4, 5]
    is_autumn = month in [9, 10, 11]

    # Define dark period ranges for each season
    summer_dark_period = list(range(21, 24)) + list(range(0, 6))  # 9 PM to 6 AM
    winter_dark_period = list(range(17, 24)) + list(range(0, 8))  # 5 PM to 8 AM
    spring_dark_period = list(range(20, 24)) + list(range(0, 6))  # 8 PM to 6 AM
    autumn_dark_period = list(range(19, 24)) + list(range(0, 7))  # 7 PM to 7 AM

    # Check if the current time falls within the dark periods
    if is_summer and hour in summer_dark_period:
        print("1")  # Light absence is not a problem in summer during these hours
        if received_data['Lux'] < 25000:
          sensor_message+="""<span style="color: #f44336;">Suboptimal Lux Levels:</span> Light levels below 20,000 lux are generally insufficient for healthy tomato plant growth, leading to poor development.<br>"""
        elif received_data['Lux'] < 70000:
          sensor_data+="""Harmful Lux Levels: Intensities above 70,000 lux can be detrimental, causing leaf burn and increased plant stress."""
    elif is_winter and hour in winter_dark_period:
        print("2")   # Light absence is a problem in winter during these hours
        if received_data['Lux'] < 25000:
          sensor_message+="""<span style="color: #f44336;">Suboptimal Lux Levels:</span> Light levels below 20,000 lux are generally insufficient for healthy tomato plant growth, leading to poor development.<br>"""
        elif received_data['Lux'] < 70000:
          sensor_data+="""Harmful Lux Levels: Intensities above 70,000 lux can be detrimental, causing leaf burn and increased plant stress."""
    elif is_spring and hour in spring_dark_period:
        print("3")   # Light absence is a problem in spring during these hours
        if received_data['Lux'] < 25000:
          sensor_message+="""<span style="color: #f44336;">Suboptimal Lux Levels:</span> Light levels below 20,000 lux are generally insufficient for healthy tomato plant growth, leading to poor development.<br>"""
        elif received_data['Lux'] < 70000:
          sensor_data+="""Harmful Lux Levels: Intensities above 70,000 lux can be detrimental, causing leaf burn and increased plant stress."""
    elif is_autumn and hour in autumn_dark_period:
        print("4")   # Light absence is a problem in autumn during these hours
        if received_data['Lux'] < 25000:
          sensor_message+="""<span style="color: #f44336;">Suboptimal Lux Levels:</span> Light levels below 20,000 lux are generally insufficient for healthy tomato plant growth, leading to poor development.<br>"""
        elif received_data['Lux'] < 70000:
          sensor_data+="""Harmful Lux Levels: Intensities above 70,000 lux can be detrimental, causing leaf burn and increased plant stress."""

    sensor_message+="</p>"




    print(sensor_message)



    # HTML content of the email / This the final result that we will send to the user.
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .header {
                background-color: #f2f2f2;
                padding: 20px;
                text-align: center;
                font-size: 35px;
                color: green; /* Change the color to green */
            }
            .content {
                margin: 20px;
                font-size: 20px;
            }
            .footer {
                background-color: #2f4f4f; /* Dark slate gray */
                padding: 10px;
                text-align: center;
                font-size: 15px;
                color: white; /* Change text color to white */
            }
        </style>
    </head>
    <body>

    <div class="header">
        <h1>AgroTips Report</h1>
    </div>

    <div class="content">
        <p>Dear User,</p>
        <p>We hope this message finds you in good health and hight spirits!!</p>
        <p>""" + sensor_message + """</p>
        <br><p>""" + image_message + """</p>
        <p><br><span style="color: green;"><strong>The latest of your plats:</strong></span></p>
        <img src="cid:image1" alt="Image" width="600">
        <br><p><i>We have worked hard in order to give you the most professional and valid assessment we can provide based on the image we sent you. However do not forget, that even our model can make mistakes!! It could be a bug on the leaf or anything. We advice to discusing disease matter with your agronomist before trying to fix the problem on your own. We are here to support them. (If you see the same disease all over again for more than 5 times and you changed the view of the camera before doing so, the possibilities that your plants carry a disease are more than 75%. In that case immidiate assistance from a professional is required).</i>

    </div>

    <div class="footer">
        <p>Thank you for using AgroTips!</p>
    </div>

    </body>
    </html>
    """

    # Create email message
    msg = EmailMessage()
    msg['From'] = email_sender
    msg['To'] = email_receiver
    msg['Subject'] = subject
    msg.set_content("The AgroTips daily report.")
    msg.add_alternative(html_content, subtype='html')




    # Attach the image as a part of the mail, it will be easier to read and more beautiful.
    with open(image_path, 'rb') as img:
        img_data = img.read()
        img_cid = 'image1'
        msg.get_payload()[1].add_related(img_data, 'image', 'jpeg', cid=img_cid)

    context = ssl.create_default_context()

    # Send the email
    with smtplib.SMTP('smtp.office365.com', 587) as smtp:
        smtp.starttls(context=context)
        smtp.login(email_sender, email_password)
        smtp.send_message(msg)
        print("Email sent successfully!")


    #Sleep for 5 hours
    time.sleep(5 * 60 * 60)
except KeyboardInterrupt:
  exit

