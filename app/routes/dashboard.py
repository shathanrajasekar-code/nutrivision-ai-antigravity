from flask import Blueprint, render_template
from flask_login import login_required, current_user
from app.models import Meal, HealthMetric

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@login_required
def index():
    meals = Meal.query.filter_by(user_id=current_user.id).order_by(Meal.timestamp.desc()).all()
    metrics = HealthMetric.query.filter_by(user_id=current_user.id).order_by(HealthMetric.date.desc()).all()
    return render_template('dashboard.html', meals=meals, metrics=metrics)
