import inspect
import os
app_path = inspect.getfile(inspect.currentframe())
module_dir = os.path.realpath(os.path.dirname(app_path))
import sys
sys.path.insert(0, module_dir)
from flask import Flask 
from flask_sqlalchemy import SQLAlchemy  
from configura import *
# import sqlalchemy as db

connectt = os.environ.get('connection')

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = connectt
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)


class Interactions(db.Model):
    __tablename__ = 'interactions'

    id = db.Column('id', db.Integer, primary_key=True) 
    user_id = db.Column('user_id', db.String())
    article_id = db.Column('article_id', db.Integer)
    interaction_type_id = db.Column('interaction_type_id', db.Integer)
    created_at = db.Column('created_at', db.DateTime)
    updated_at = db.Column('updated_at', db.DateTime)
