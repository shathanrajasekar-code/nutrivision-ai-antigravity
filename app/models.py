from app import db, login_manager
from flask_login import UserMixin
from datetime import datetime

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(60), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    weight = db.Column(db.Float, nullable=True)  # in kg
    height = db.Column(db.Float, nullable=True)  # in cm
    health_goals = db.Column(db.String(200), nullable=True)  # e.g. weight loss, muscle gain

    meals = db.relationship('Meal', backref='author', lazy=True)
    chat_history = db.relationship('ChatHistory', backref='user', lazy=True)
    health_metrics = db.relationship('HealthMetric', backref='user', lazy=True)

class Meal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    total_calories = db.Column(db.Float, nullable=True)
    health_score = db.Column(db.Integer, nullable=True)        # 0–100 ANTIGRAVITY score
    health_status = db.Column(db.String(50), nullable=True)    # Food category (e.g. Fast Food, Meal, Vegetable)
    analysis_insights = db.Column(db.Text, nullable=True)      # Kept for legacy display

    # ANTIGRAVITY ADDITIONS
    food_detected = db.Column(db.String(100), nullable=True)   # Top detected food class name
    food_confidence = db.Column(db.String(10), nullable=True)  # Confidence string e.g. "94.2%"
    genai_result_json = db.Column(db.Text, nullable=True)      # Full JSON from genai_service.evaluate_meal()

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    detections = db.relationship('FoodDetection', backref='meal', lazy=True)

class FoodDetection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    food_name = db.Column(db.String(100), nullable=False)
    portion_size = db.Column(db.Float, nullable=True)          # grams or units
    confidence = db.Column(db.Float, nullable=True)
    meal_id = db.Column(db.Integer, db.ForeignKey('meal.id'), nullable=False)

    nutrition = db.relationship('NutritionData', backref='detection', uselist=False)

class NutritionData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    calories = db.Column(db.Float, nullable=False)
    protein = db.Column(db.Float, nullable=False)
    carbs = db.Column(db.Float, nullable=False)
    fats = db.Column(db.Float, nullable=False)
    sugar = db.Column(db.Float, nullable=False)
    fiber = db.Column(db.Float, nullable=False)
    food_detection_id = db.Column(db.Integer, db.ForeignKey('food_detection.id'), nullable=False)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(10), nullable=False)            # 'user' or 'bot'
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class HealthMetric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    weight = db.Column(db.Float, nullable=True)
    daily_calories = db.Column(db.Float, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
