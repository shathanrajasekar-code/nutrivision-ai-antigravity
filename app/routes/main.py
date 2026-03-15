from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/features')
def features():
    return render_template('features.html')

@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    # Placeholder for contact form processing
    return render_template('contact.html')
