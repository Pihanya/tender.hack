from flask import Flask, request, render_template
from flask_socketio import SocketIO
import time

import script 


app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)
timeout_seconds = 30

@app.route('/query')
def query_example():
    return 'Todo...'

@app.route('/')
def hello_world():
    return render_template('index.html')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

def get_simple_answer(message):
    print("Start working on message", message)
    return str(script.get_simple_answer(message))
#     response_message = ''
#     start_time = time.clock()
#     max_metric, max_sum = None, 0
#     # проходимся по всей базе вопросов
#     for question in questions:
#         cur_time = time.clock()
#         if cur_time - start_time > timeout_seconds:
#             return response_message
        
#         metric, hits = find_metric(message, question)
#         if sum(metric[:4]) > max_sum:
#             max_sum = sum(metric[:4])
#             max_metric = metric
    
#     metrics, hits = max_metric
    
    
#     return message.upper()

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: '+ str(json))
    json['chanse'] = -1
    socketio.emit('my response', json, callback=messageReceived)
    
    answer, request, tokens, metric = get_simple_answer(json['message'])
    json['answer'] =  answer
    json['keywords'] = tokens
    json['username'] = 'bot'
    json['metric'] = metric
    socketio.emit('my response', json, callback=messageReceived)

    #json['message'] = get_simple_answer(json['message'])
    #socketio.emit('my response', json, callback=messageReceived)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug=True)

