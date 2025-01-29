from sqlalchemy import create_engine, Column, Integer, String, Float, Text, JSON, ForeignKey, TIMESTAMP, func
from sqlalchemy.orm import sessionmaker, scoped_session, relationship
from sqlalchemy.ext.declarative import declarative_base


# Database configuration
DATABASE_URL =
engine = create_engine(DATABASE_URL, echo=False)
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)
Base = declarative_base()

class Book(Base):
    __tablename__ = 'books'
    isbn13 = Column(String(20), primary_key=True)
    title = Column(String(255), nullable=True)
    author = Column(String(100), nullable=True)
    date_published = Column(Text, nullable=True)
    genre = Column(Text, nullable=True)
    average_review_score = Column(Text, nullable=True)
    normalized_average_review_score = Column(Float, nullable=True)
    length = Column(Integer, nullable=True)
    encoded_genre = Column(JSON, nullable=True)
    encoded_author = Column(Integer, nullable=True)
    summary = Column(Text, nullable=True)
    cover_image_url = Column(String(255), nullable=True)
    rating_count = Column(Integer, nullable=True)
    normalized_rating_count = Column(Float, nullable=True)
    normalized_review_count = Column(Float, nullable=True)
    review_count = Column(Integer, nullable=True)
    image_location = Column(String(255), nullable=True)
    preprocessed_summary = Column(Text, nullable=True)
    preprocessed_title = Column(Text, nullable=True)
    reviews_per_year = Column(Float, nullable=True)
    ratings_per_year = Column(Float, nullable=True)
    normalized_reviews_per_year = Column(Float, nullable=True)
    normalized_ratings_per_year = Column(Float, nullable=True)


    cover_colors = relationship("CoverColorData", back_populates="book", cascade="save-update, merge")
    image_quality = relationship("ImageQuality", back_populates="book", uselist=False, cascade="save-update, merge")

class CoverColorData(Base):
    __tablename__ = 'cover_color_data'
    isbn13 = Column(String(20), ForeignKey('books.isbn13'), primary_key=True)
    color_1_rgb = Column(String(20))
    color_1_pct = Column(Float)
    color_1_name = Column(String(20))
    color_2_rgb = Column(String(20))
    color_2_pct = Column(Float)
    color_2_name = Column(String(20))
    color_3_rgb = Column(String(20))
    color_3_pct = Column(Float)
    color_3_name = Column(String(20))
    color_4_rgb = Column(String(20))
    color_4_pct = Column(Float)
    color_4_name = Column(String(20))
    color_5_rgb = Column(String(20))
    color_5_pct = Column(Float)
    color_5_name = Column(String(20))

    book = relationship("Book", back_populates="cover_colors")


class ImageQuality(Base):
    __tablename__ = 'image_quality'
    isbn13 = Column(String(20), ForeignKey('books.isbn13'), primary_key=True)
    sharpness = Column(Float, nullable=True)
    contrast = Column(Float, nullable=True)
    saturation = Column(Float, nullable=True)
    noise = Column(Float, nullable=True)
    entropy = Column(Float, nullable=True)
    book = relationship("Book", back_populates="image_quality")

class GenreTierStatistics(Base):
    __tablename__ = 'genre_tier_statistics'
    id = Column(Integer, primary_key=True, autoincrement=True)
    genre = Column(String(255), nullable=False)
    tier = Column(Integer, nullable=False)
    books_in_tier = Column(Integer, nullable=False)
    total_books_in_genre = Column(Integer, nullable=False)
    active_books = Column(Integer, nullable=False)
    tier_min_score = Column(Float, nullable=False)
    tier_max_score = Column(Float, nullable=False)
    ratings_per_year_mean = Column(Float, nullable=False)
    ratings_per_year_std = Column(Float, nullable=False)
    reviews_per_year_mean = Column(Float, nullable=False)
    reviews_per_year_std = Column(Float, nullable=False)
    avg_rating_mean = Column(Float, nullable=False)
    avg_rating_std = Column(Float, nullable=False)
    mean_rating_count = Column(Float, nullable=False)
    mean_review_count = Column(Float, nullable=False)
    percentile_lower = Column(Float, nullable=False)
    percentile_upper = Column(Float, nullable=False)
    last_updated = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


# Utility function to get a session
def get_session():
    return Session()

# Create database tables based on the Base metadata
Base.metadata.create_all(engine)

#from common.py import get_session, Book