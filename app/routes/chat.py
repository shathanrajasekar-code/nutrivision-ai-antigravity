import json
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from app.models import ChatHistory, Meal
from app import db
from app.services.chatbot_service import get_chatbot_response

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['GET'])
@login_required
def chat():
    history = ChatHistory.query.filter_by(user_id=current_user.id)\
                               .order_by(ChatHistory.timestamp.asc()).all()
    return render_template('chatbot.html', history=history)


@chat_bp.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # ── Fetch last-scanned meal context for ANTIGRAVITY injection ──────────
    last_meal = None
    latest_meal = (
        Meal.query.filter_by(user_id=current_user.id)
                  .order_by(Meal.timestamp.desc())
                  .first()
    )
    if latest_meal and latest_meal.genai_result_json:
        try:
            last_meal = json.loads(latest_meal.genai_result_json)
        except (json.JSONDecodeError, TypeError):
            last_meal = None

    # ── Persist user message ───────────────────────────────────────────────
    user_chat = ChatHistory(role='user', message=user_message, user_id=current_user.id)
    db.session.add(user_chat)

    history_objs = (
        ChatHistory.query.filter_by(user_id=current_user.id)
                         .order_by(ChatHistory.timestamp.asc()).all()
    )
    bot_response = get_chatbot_response(user_message, history_objs, last_meal=last_meal)

    bot_chat = ChatHistory(role='bot', message=bot_response, user_id=current_user.id)
    db.session.add(bot_chat)
    db.session.commit()

    return jsonify({'response': bot_response}), 200
