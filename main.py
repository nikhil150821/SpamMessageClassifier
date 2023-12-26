from flask import Flask, render_template
from streamlit import ScriptRequestQueue, forward_msg

app = Flask(__name__)

# Initialize Streamlit ScriptRequestQueue
st_queue = ScriptRequestQueue()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/streamlit_app', methods=['GET', 'POST'])
def streamlit_app():
    if st_queue:
        forward_msg(msg=st_queue.get_script_request())
    return ''

if __name__ == '__main__':
    app.run(debug=True)
