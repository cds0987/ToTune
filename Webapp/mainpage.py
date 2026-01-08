from flask import Blueprint, render_template
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session
mainpage_bth = Blueprint('mainpage', __name__)
@mainpage_bth.route("/")
def usermainpage():
    # Render the template for the user main page
    return render_template("index.html")