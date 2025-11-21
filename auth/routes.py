import json
import os
from datetime import datetime
from collections import defaultdict

from flask import request, jsonify, render_template, redirect, url_for, session, flash, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash

from auth import auth_bp
from models.user import db, User
from shared_utils import Config

# Rate limiting storage (in production, use Redis)
login_attempts = defaultdict(list)
MAX_LOGIN_ATTEMPTS = 5
RATE_LIMIT_WINDOW = 300  # 5 minutes

def _load_group_mapping():
    raw = os.getenv("SSO_GROUP_MAP", "").strip()
    if not raw:
        return {}
    try:
        mapping = json.loads(raw)
        if isinstance(mapping, dict):
            return mapping
    except Exception:
        pass
    return {}

_SSO_GROUP_MAP = _load_group_mapping()


def _resolve_groups_for_email(email: str) -> list[str]:
    email = (email or "").strip().lower()
    if not email:
        return [Config.ORACLE_DEFAULT_GROUP]

    groups = set()
    # Exact email match
    direct = _SSO_GROUP_MAP.get(email)
    if isinstance(direct, (list, tuple)):
        groups.update(str(g).strip().lower() for g in direct if str(g).strip())

    # Domain wildcard mapping using "@domain" keys
    if "@" in email:
        domain_key = f"@{email.split('@', 1)[1]}"
        domain_groups = _SSO_GROUP_MAP.get(domain_key)
        if isinstance(domain_groups, (list, tuple)):
            groups.update(str(g).strip().lower() for g in domain_groups if str(g).strip())

    if not groups:
        # Basic defaults: admins get elevated access, others receive public access
        if email.endswith("@admin.local") or email.startswith("admin"):
            groups.update({"oracle.admin", "oracle.hr"})
        else:
            groups.add(Config.ORACLE_DEFAULT_GROUP)

    return sorted(groups)

def is_rate_limited(identifier):
    """Check if identifier (IP or email) is rate limited"""
    now = datetime.utcnow()
    attempts = login_attempts[identifier]
    
    # Remove attempts older than the window
    attempts[:] = [attempt for attempt in attempts if (now - attempt).seconds < RATE_LIMIT_WINDOW]
    
    return len(attempts) >= MAX_LOGIN_ATTEMPTS

def record_login_attempt(identifier):
    """Record a login attempt"""
    login_attempts[identifier].append(datetime.utcnow())

@auth_bp.route('/login', methods=['GET'])
def login():
    """Render login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('auth/login.html')

@auth_bp.route('/login', methods=['POST'])
def login_post():
    """Handle login form submission"""
    data = request.get_json() if request.is_json else request.form
    
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    # Rate limiting
    client_ip = request.remote_addr
    if is_rate_limited(client_ip) or is_rate_limited(email):
        return jsonify({'error': 'Too many login attempts. Please try again later.'}), 429
    
    if not email or not password:
        record_login_attempt(client_ip)
        return jsonify({'error': 'Email and password are required'}), 400
    
    user = User.query.filter_by(email=email).first()
    
    if not user or not user.check_password(password):
        record_login_attempt(client_ip)
        record_login_attempt(email)
        return jsonify({'error': 'Invalid email or password'}), 401

    login_user(user, remember=True)
    session['oracle_groups'] = _resolve_groups_for_email(user.email)
    session['oracle_role'] = user.role

    return jsonify({
        'ok': True,
        'role': user.role,
        'email': user.email,
        'message': 'Login successful'
    })

@auth_bp.route('/register', methods=['GET'])
def register():
    """Render registration page"""
    if not os.getenv('ALLOW_SIGNUP', 'true').lower() == 'true':
        return jsonify({'error': 'Registration is disabled'}), 403
    
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    return render_template('auth/register.html')

@auth_bp.route('/register', methods=['POST'])
def register_post():
    """Handle registration form submission"""
    if not os.getenv('ALLOW_SIGNUP', 'true').lower() == 'true':
        return jsonify({'error': 'Registration is disabled'}), 403
    
    data = request.get_json() if request.is_json else request.form
    
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    name = data.get('name', '').strip()
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters long'}), 400
    
    # Check if user already exists
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    # Create new user
    user = User(
        email=email,
        role='user'
    )
    user.set_password(password)
    
    try:
        db.session.add(user)
        db.session.commit()
        return jsonify({'ok': True, 'message': 'Registration successful'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@auth_bp.route('/logout', methods=['POST', 'GET'])
def logout():
    """Handle logout (supports POST via fetch and GET via link fallback)"""
    # Only attempt to logout if authenticated; otherwise behave like successful logout
    if current_user.is_authenticated:
        logout_user()
    session.clear()

    # POST requests (from tests or fetch) receive JSON 200
    if request.method == 'POST':
        # Explicitly clear cookies to avoid sticky sessions across hosts
        resp = jsonify({'ok': True, 'message': 'Logged out successfully'})
        # session cookie
        try:
            resp.delete_cookie(current_app.session_cookie_name, path='/', samesite=current_app.config.get('SESSION_COOKIE_SAMESITE', 'Lax'))
        except Exception:
            pass
        # remember me cookie
        try:
            resp.delete_cookie('remember_token', path='/')
        except Exception:
            pass
        return resp

    # Regular navigation: send user to login page
    resp = redirect(url_for('auth.login'))
    try:
        resp.delete_cookie(current_app.session_cookie_name, path='/', samesite=current_app.config.get('SESSION_COOKIE_SAMESITE', 'Lax'))
    except Exception:
        pass
    try:
        resp.delete_cookie('remember_token', path='/')
    except Exception:
        pass
    return resp

@auth_bp.route('/me', methods=['GET'])
def me():
    """Get current user info"""
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'role': current_user.role,
            'email': current_user.email,
            'security_groups': session.get('oracle_groups', [])
        })
    else:
        return jsonify({
            'authenticated': False,
            'role': None,
            'email': None,
            'security_groups': []
        })
