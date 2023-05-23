from flask import Flask, request
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)

@app.route('/api/execute-script', methods=['POST'])
def execute_script():
    # Retrieve any data from the request if needed
    # ...

    # Execute your Python script here
    # ...
    
    os.system('python3 /Users/ml/Desktop/ILS/integration/test_execution.py')
    # Return a response if desired
    return 'Script executed successfully'
    #return data_list

if __name__ == '__main__':
    app.run()


    



