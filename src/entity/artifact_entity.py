# artifact_entity.py
"""
Industry-grade artifact entity definitions for STOCKER pipeline.
Includes type hints, docstrings, versioning, run metadata, error handling, and serialization.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import json
import yaml
import pandas as pd
import hashlib

class ArtifactStatus(str, Enum):
    CREATED = "created"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class IngestionArtifact:
    """Artifact for raw data ingestion step."""
    symbols: List[str]
    raw_data_paths: List[str]
    status: ArtifactStatus
    run_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    data_transformation: Optional[Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: Optional[Dict[str, str]] = field(default_factory=dict)
    def to_dict(self):
        return asdict(self)
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
    def to_yaml(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

@dataclass
class NewsArtifact:
    """Artifact for news ingestion step."""
    symbol: str
    articles: List[Dict]
    status: ArtifactStatus
    run_id: Optional[str] = None
    data_transformation: Optional[Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: Optional[List[str]] = field(default_factory=list)
    def to_dict(self):
        return asdict(self)
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
    def to_yaml(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

@dataclass
class ModelArtifact:
    """Artifact for model training/saving step."""
    model_path: str
    metrics_path: Optional[str] = None
    model_version: Optional[str] = None
    status: ArtifactStatus = ArtifactStatus.CREATED
    run_id: Optional[str] = None
    created_at: Optional[str] = None
    data_transformation: Optional[Dict[str, Any]] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    errors: Optional[List[str]] = field(default_factory=list)
    def to_dict(self):
        return asdict(self)
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
    def to_yaml(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

@dataclass
class ReportArtifact:
    """Artifact for reporting step."""
    report_path: str
    generated_at: str
    status: ArtifactStatus = ArtifactStatus.SUCCESS
    run_id: Optional[str] = None
    data_transformation: Optional[Dict[str, Any]] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    errors: Optional[List[str]] = field(default_factory=list)
    def to_dict(self):
        return asdict(self)
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
    def to_yaml(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

@dataclass
class PredictionArtifact:
    """Artifact for prediction/inference step."""
    predictions_path: str
    status: ArtifactStatus
    run_id: Optional[str] = None
    completed_at: Optional[str] = None
    data_transformation: Optional[Dict[str, Any]] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    errors: Optional[List[str]] = field(default_factory=list)
    def to_dict(self):
        return asdict(self)
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
    def to_yaml(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

@dataclass
class ValidationArtifact:
    step: str
    status: str  # 'success', 'fail', 'warning'
    errors: Optional[list] = field(default_factory=list)
    warnings: Optional[list] = field(default_factory=list)
    drift_metrics: Optional[dict] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data_hash: Optional[str] = None
    context: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self):
        return {
            "step": self.step,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "drift_metrics": self.drift_metrics,
            "timestamp": self.timestamp,
            "data_hash": self.data_hash,
            "context": self.context
        }

    @staticmethod
    def hash_dataframe(df):
        # Hash the DataFrame for traceability
        return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

@dataclass
class PipelineRunArtifact:
    """Top-level artifact for a pipeline run, linking all step artifacts and logs."""
    run_id: str
    started_at: str
    completed_at: Optional[str] = None
    status: ArtifactStatus = ArtifactStatus.CREATED
    config_path: Optional[str] = None
    log_path: Optional[str] = None
    data_transformation: Optional[Dict[str, Any]] = field(default_factory=dict)
    ingestion: Optional[IngestionArtifact] = None
    model: Optional[ModelArtifact] = None
    report: Optional[ReportArtifact] = None
    prediction: Optional[PredictionArtifact] = None
    validation: Optional[ValidationArtifact] = None
    extra: Optional[Dict[str, Any]] = field(default_factory=dict)
    def to_dict(self):
        return asdict(self)
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
    def to_yaml(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

# Add more artifact entities or extend as needed for your workflow.
