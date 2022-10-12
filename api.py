from flask import Flask
import json
import random
app = Flask(__name__)
@app.route('/', methods=["GET"])
def getFruit():
    listFruit= ['apple', 'banana','orange']
    choiceFruit = listFruit[random.randint(0,2)]
    return json.dumps({"fruit":choiceFruit})
    
if __name__ == '__main__':
    app.run(debug=True)
print('api start')