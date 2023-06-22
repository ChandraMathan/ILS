
import os
import pickle
import json

#load data or define the steps

# Open the file in read mode
with open('/Users/ml/Downloads/data_poc.json', 'r') as file:
    content = file.read()

# Print the content of the file


process_steps = json.loads(content)


#write test cases and save as pickle
instruction_dir = 'process_steps.pkl'


process_steps = [
                "navigate to url:http://localhost:3000/webtrain1", "enter 'First name test' in field 'First Name'", "enter 'AUS' in field 'Country of Birth'",
                 "navigate to url:http://localhost:3000/webtrain5", "enter 'UTS' in field 'Institution Name'", "enter 'Computer Science' in field 'Course Name'"
                 ]



#pre processing instructions: split in to instructions and data 

instruction_data = {}
for index in range(len(process_steps)):
    instruction_data[index] = {}
    if "navigate to url:" in process_steps[index]:
        instruction_data[index]['process'] = 'navigate to url'
        instruction_data[index]['data'] = process_steps[index].replace('navigate to url:',"")
    else:

        instruction_data[index]['process'] = 'enter data in'
        instruction_data[index]['data'] = {}
        

        pos = [pos for pos, char in enumerate(process_steps[index]) if char == "'"]
        instruction_data[index]['data']['field name'] = process_steps[index][pos[2]:pos[3]+1].strip("'")
        instruction_data[index]['data']['value'] = process_steps[index][pos[0]:pos[1]+1].strip("'")

#print(instruction_data)

with open(instruction_dir, 'wb') as f:
            pickle.dump(instruction_data, f)

os.system('python3 /Users/ml/Desktop/ILS/integration/e2e_integration_py.py')
