from flask import Flask, request
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)

@app.route('/api/execute-script', methods=['POST'])
def execute_script():
    
    os.system('python3 /Users/ml/Desktop/ILS/integration/test_execution.py')
    return 'Script executed successfully'
    

if __name__ == '__main__':
    app.run()


    



