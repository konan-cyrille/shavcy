from flask import render_template, redirect, url_for, request
from pdapp import pdapp
import sys
from labelImg.labelImg import main

@pdapp.route('/', methods=['GET', 'POST'])
@pdapp.route('/index')
def index():

    return render_template('index.html', title='Home')

@pdapp.route("/annotation/", methods=['GET', 'POST'])
def labelisation():
    main()