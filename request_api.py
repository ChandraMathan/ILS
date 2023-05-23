import requests

url = 'http://127.0.0.1:5000/api/execute-script'

x = requests.post(url, verify=False)

print(x.text)
