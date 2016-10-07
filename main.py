from os import path
import os
from flask import Flask
from flask import render_template
from flask import request

from dateutil.parser import parse as date_parse

from random import randint

from coint import run_coint


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def calc():
    # get post params
    symbols = [s.strip() for s in request.form['symbols'].split(",")]
    date = date_parse(request.form['date'])

    # generate random request id
    request_id = randint(0, 10000000000)
    loc = os.path.join("static", str(request_id))

    if os.path.exists(loc):
        os.removedirs(loc)
    os.makedirs(loc)

    result = run_coint(symbols, date, loc)

    return render_template('result.html', symbols=symbols, date=date, coint_heatmap=loc+"/heatmap.png",
                           summary=result[0], plots=result[1])


# @app.route('/test')
# def test():
#     return "ok"

app.run(debug=True, host='0.0.0.0')