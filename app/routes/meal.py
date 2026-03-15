import os
import uuid
import json
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models import Meal, FoodDetection, NutritionData
from app.services.cv_service import detect_food
from app.services.nutrition_service import estimate_nutrition
from app.services.genai_service import evaluate_meal

meal_bp = Blueprint('meal', __name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@meal_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image part', 'danger')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            # ── AI Processing Pipeline ──────────────────────────────────────
            detections = detect_food(filepath)
            nutrition_results = estimate_nutrition(detections)

            # Extract top YOLO confidence if available
            top_confidence = None
            if detections:
                first_det = detections[0] if isinstance(detections, list) else None
                if first_det and isinstance(first_det, dict):
                    top_confidence = first_det.get('confidence')

            genai_evaluation = evaluate_meal(nutrition_results, yolo_confidence=top_confidence)

            # ── Persist to Database ─────────────────────────────────────────
            insights_str = "\n".join(
                [f"- {i}" for i in genai_evaluation.get('insights', [])]
            )
            opt = genai_evaluation.get('optimisation_tip', '')
            if opt:
                insights_str += f"\n\nOptimization Tip: {opt}"

            meal = Meal(
                image_path=unique_filename,
                total_calories=genai_evaluation.get('macros', {}).get('calories', 0),
                health_score=genai_evaluation.get('health_score', 0),
                health_status=genai_evaluation.get('category', 'Unknown'),
                analysis_insights=insights_str,
                # ANTIGRAVITY structured fields
                food_detected=genai_evaluation.get('food_detected', ''),
                food_confidence=genai_evaluation.get('confidence', ''),
                genai_result_json=json.dumps(genai_evaluation),
                user_id=current_user.id
            )
            db.session.add(meal)
            db.session.flush()  # get meal.id

            for f_name, n_data in nutrition_results.items():
                fd = FoodDetection(
                    food_name=f_name,
                    portion_size=n_data.get('portion', 100),
                    confidence=n_data.get('confidence', 0.9),
                    meal_id=meal.id
                )
                db.session.add(fd)
                db.session.flush()

                nd = NutritionData(
                    calories=genai_evaluation.get('macros', {}).get('calories', 0),
                    protein=genai_evaluation.get('macros', {}).get('protein_g', 0),
                    carbs=genai_evaluation.get('macros', {}).get('carbs_g', 0),
                    fats=genai_evaluation.get('macros', {}).get('fat_g', 0),
                    sugar=0,
                    fiber=genai_evaluation.get('macros', {}).get('fibre_g', 0),
                    food_detection_id=fd.id
                )
                db.session.add(nd)

            db.session.commit()
            return redirect(url_for('meal.result', meal_id=meal.id))

    return render_template('upload.html')


@meal_bp.route('/result/<int:meal_id>')
@login_required
def result(meal_id):
    meal = Meal.query.get_or_404(meal_id)
    if meal.user_id != current_user.id:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard.index'))

    # Deserialise the full ANTIGRAVITY evaluation for the template
    eval_data = {}
    if meal.genai_result_json:
        try:
            eval_data = json.loads(meal.genai_result_json)
        except (json.JSONDecodeError, TypeError):
            eval_data = {}

    return render_template('result.html', meal=meal, eval_data=eval_data)
