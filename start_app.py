import eventlet
eventlet.monkey_patch()

from cortex import create_app

app = create_app()

from cortex.extensions import socketio

if __name__ == '__main__':
    socketio.run(app, port=5000, host='0.0.0.0', debug=True)