from flask import Flask, render_template, request
import os
import tempfile
import subprocess
import json
import sys

def process_input(input_str):
    # Find the start and end indices of the JSON data
    start_index = input_str.find('{')
    end_index = input_str.rfind('}') + 1

    # Extract the JSON data
    json_data = input_str[start_index:end_index]
    # Parse the input string
    data = json.loads(json_data)

    # Extract the probabilities list and prediction
    probs = data["probs"][0]
    prediction = data["prediction"]

    # Find the index of the maximum probability
    max_prob_index = probs.index(max(probs))

    # Create a new dictionary with only the maximum probability and prediction
    result = {
        "max_prob": round(probs[max_prob_index],3),
        "prediction": prediction
    }

    return result

adress = sys.argv[1]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    # Save the file to a temporary location
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)
    
    #print("The received argument variable is: ", adress)
    command = f"curl -X POST -H 'Content-Type: multipart/form-data' -H 'Accept: application/json' -F 'beans_img=@{file_path}' http://{adress}:4000/make_prediction"
    # Get the classification result
    response = subprocess.getoutput(command)

    # Convert the response to a JSON object
    result = process_input(response)

    # Render the results page
    return render_template('result.html', result=result)
    
if __name__ == '__main__':
    app.run(debug=True)