from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

pdapp = Flask(__name__)
pdapp.config.from_object(Config)
db = SQLAlchemy(pdapp)
migrate = Migrate(pdapp, db)

from pdapp import views, models