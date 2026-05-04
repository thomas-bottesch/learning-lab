"""Verify all three pipelines compile to valid KFP YAML without errors."""
import tempfile
from kfp import compiler
from pipelines.training_pipeline import training_pipeline
from pipelines.retraining_pipeline import retraining_pipeline
from pipelines.evaluation_pipeline import evaluation_pipeline


def test_training_pipeline_compiles():
    with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        compiler.Compiler().compile(training_pipeline, f.name)


def test_retraining_pipeline_compiles():
    with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        compiler.Compiler().compile(retraining_pipeline, f.name)


def test_evaluation_pipeline_compiles():
    with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        compiler.Compiler().compile(evaluation_pipeline, f.name)
