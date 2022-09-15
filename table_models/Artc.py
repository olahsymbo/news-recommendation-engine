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

# Retrieve set environment variables
connectt = os.environ.get('connection')

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = connectt
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)


article_category = db.Table('article_category',
    db.Column('article_id', db.Integer, db.ForeignKey('articles.id')),
    db.Column('category_id', db.Integer, db.ForeignKey('categories.id'))               
)


class Articles(db.Model):
    __tablename__ = 'articles'

    id = db.Column('id', db.Integer, primary_key=True) 
    title = db.Column('title', db.String())
    author = db.Column('author', db.String())
    summary = db.Column('summary', db.Text)
    content = db.Column('content', db.Text)
    news_source_id = db.Column('news_source_id', db.Integer)
    aggregator = db.Column('aggregator', db.String)
    url = db.Column('url', db.String())
    status_id = db.Column('status_id', db.Integer)
    published_date = db.Column('published_date', db.DateTime)
    created_at = db.Column('created_at', db.DateTime)
    updated_at = db.Column('updated_at', db.DateTime)
    deleted_at = db.Column('deleted_at', db.DateTime)
    additional_data = db.Column('additional_data', db.Integer)
    
    art_category = db.relationship('Categories',
                                   secondary = article_category,
                                   backref= db.backref('news', lazy='dynamic'))


class Categories(db.Model):
    __tablename__ = 'categories'
    
    id = db.Column('id', db.Integer, primary_key=True)
    # article_id = db.Column(db.Integer, db.ForeignKey('articles.id'))
    name = db.Column('name', db.String())