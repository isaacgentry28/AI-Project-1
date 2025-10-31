from dataclasses import dataclass
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash
from config import DB_PATH


Base = declarative_base()
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@dataclass
class User(Base):
 __tablename__ = 'users'
id: int = Column(Integer, primary_key=True)
email: str = Column(String(255), unique=True, nullable=False)
password_hash: str = Column(String(255), nullable=False)


def set_password(self, password: str):

 self.password_hash = generate_password_hash(password)


def check_password(self, password: str) -> bool:
 return check_password_hash(self.password_hash, password)




def init_db():
 Base.metadata.create_all(bind=engine)