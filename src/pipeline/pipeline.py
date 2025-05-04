# pipeline.py
"""
Robust, industry-grade pipeline orchestrator for STOCKER.
- Pluggable step registry for extensibility
- Unified artifact, log, and metadata management
- Run directory per execution (timestamped)
- CLI/config overrides
- Experiment tracking and notification hooks (placeholders)
"""
import argparse
import logging
import os
import sys
import traceback
import json
from datetime import datetime
from src.entity.config_entity import StockerConfig
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import get_advanced_logger

# --- Utility: create run directory ---
def create_run_dir(base_dir="runs"):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id

# --- Utility: save metadata ---
def save_metadata(run_dir, config, mode, run_id):
    meta = {
        "run_id": run_id,
        "mode": mode,
        "config": config.__dict__,
        # Add more metadata as needed (git hash, env info, etc.)
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

# --- Step registry ---
PIPELINE_STEPS = {
    "train": TrainingPipeline,
    "predict": PredictionPipeline,
    # Easily add more steps here ("evaluate": EvaluatePipeline, ...)
}

class StockerPipeline:
    def __init__(self, config: StockerConfig, mode: str = "train", run_dir: str = None):
        self.config = config
        self.mode = mode
        self.run_dir = run_dir or create_run_dir()[0]
        self.logger = get_advanced_logger("main_pipeline", log_to_file=True, log_dir=self.run_dir)

    def run(self):
        self.logger.info(f"Starting StockerPipeline in '{self.mode}' mode. Artifacts/logs in {self.run_dir}")
        save_metadata(self.run_dir, self.config, self.mode, os.path.basename(self.run_dir))
        step_cls = PIPELINE_STEPS.get(self.mode)
        if not step_cls:
            self.logger.error(f"Unknown pipeline mode: {self.mode}")
            raise ValueError(f"Unknown pipeline mode: {self.mode}")
        try:
            pipeline = step_cls(self.config)
            artifacts = pipeline.run()
            # Save pipeline artifacts (summary only; details saved by step)
            with open(os.path.join(self.run_dir, "artifacts_summary.json"), "w") as f:
                json.dump({k: str(v) for k,v in artifacts.items()}, f, indent=2)
            self.logger.info(f"{self.mode.capitalize()} pipeline completed.")
            # --- Placeholder: experiment tracking, notification hooks ---
            # e.g., mlflow.log_artifacts(self.run_dir), send_slack_notification(...)
            return artifacts
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
            # Optionally: send failure notification here
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stocker ML pipeline.")
    parser.add_argument("--mode", choices=list(PIPELINE_STEPS.keys()), default="train", help="Pipeline mode: train or predict")
    parser.add_argument("--config", default=None, help="Path to config file (optional)")
    args = parser.parse_args()
    config = StockerConfig()  # TODO: load from args.config if provided
    run_dir, _ = create_run_dir()
    pipeline = StockerPipeline(config, mode=args.mode, run_dir=run_dir)
    pipeline.run()
