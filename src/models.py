from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_login import UserMixin

db = SQLAlchemy()

# Environmental impact data (in kg CO2 saved per item)
ENVIRONMENTAL_IMPACT = {
    'battery': 0.05,
    'biological': 0.02,
    'clothes': 0.15,
    'glass': 0.08,
    'metal': 0.12,
    'paper': 0.03,
    'plastic': 0.07,
    'shoes': 0.20,
    'trash': 0.01
}

# Points system
POINTS_SYSTEM = {
    'classification': 2,
    'verified_post': 3,
    'challenge_completion': 10,
    'daily_streak': 1
}

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    streak = db.Column(db.Integer, default=0)
    total_points = db.Column(db.Integer, default=0)
    weekly_points = db.Column(db.Integer, default=0)
    co2_saved = db.Column(db.Float, default=0.0)
    waste_classified = db.Column(db.Integer, default=0)
    verified_posts = db.Column(db.Integer, default=0)
    bio = db.Column(db.String(200), default="")
    badges = db.Column(db.Text, default="")
    streak_freezes = db.Column(db.Integer, default=0)
    
    # Relationships
    posts = db.relationship('Post', backref='author', lazy=True)
    classifications = db.relationship('Classification', backref='user', lazy=True)
    challenges = db.relationship('UserChallenge', backref='user', lazy=True)
    comments = db.relationship('Comment', backref='author', lazy=True)
    likes = db.relationship('Like', backref='user', lazy=True)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(20), nullable=False)  # eco_tips, waste_facts, diy_reuse
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    likes_count = db.Column(db.Integer, default=0)
    comments_count = db.Column(db.Integer, default=0)
    is_verified = db.Column(db.Boolean, default=False)
    reported_count = db.Column(db.Integer, default=0)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    comments = db.relationship('Comment', backref='post', lazy=True)
    likes = db.relationship('Like', backref='post', lazy=True)

class Challenge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    target_count = db.Column(db.Integer, default=0)
    reward_points = db.Column(db.Integer, default=10)
    badge_name = db.Column(db.String(50), default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user_challenges = db.relationship('UserChallenge', backref='challenge', lazy=True)

class UserChallenge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    challenge_id = db.Column(db.Integer, db.ForeignKey('challenge.id'), nullable=False)
    progress = db.Column(db.Integer, default=0)
    is_completed = db.Column(db.Boolean, default=False)
    completed_at = db.Column(db.DateTime, nullable=True)
    points_earned = db.Column(db.Integer, default=0)
    
    # Ensure a user can only join a challenge once
    __table_args__ = (db.UniqueConstraint('user_id', 'challenge_id', name='unique_user_challenge'),)

class Classification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    waste_category = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    co2_saved = db.Column(db.Float, default=0.0)
    points_earned = db.Column(db.Integer, default=0)
    classified_at = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(200), nullable=True)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_reported = db.Column(db.Boolean, default=False)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)

class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    
    # Ensure a user can only like a post once
    __table_args__ = (db.UniqueConstraint('user_id', 'post_id', name='unique_user_post_like'),)