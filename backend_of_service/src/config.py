import os
from pathlib import Path


class Config:
    """Config for application."""
    backend_host: str = "0.0.0.0"
    backend_port: int = 80
    current_dir: Path = Path(__file__).resolve().parent.parent
    model_path: str = os.path.join(current_dir, 'model')
