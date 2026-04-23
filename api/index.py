import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# For Vercel serverless
from mangum import Mangum

handler = Mangum(app, lifespan="off")
