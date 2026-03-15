from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, current_user, logout_user, login_required
from app import db, bcrypt
from app.models import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        age = request.form.get('age')
        weight = request.form.get('weight')
        height = request.form.get('height')
        goals = request.form.get('goals')
        
        # Validation
        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Email already registered.', 'danger')
            return redirect(url_for('auth.signup'))
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password_hash=hashed_password,
                    age=age, weight=weight, height=height, health_goals=goals)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('auth.login'))
        
    return render_template('signup.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user, remember=request.form.get('remember'))
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard.index'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('main.index'))
