#!/bin/bash
# Script to manually run the LakeFS initial data setup pipeline
# This is useful for testing outside of the CI/CD workflow

set -e

echo "========================================="
echo "LakeFS Initial Data Setup Pipeline"
echo "========================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "create_initial_data_pipeline.py" ]; then
    echo "Error: Please run this script from the example_git_project directory"
    exit 1
fi

# Step 1: Setup Python environment
echo "Step 1: Setting up Python environment..."
python3.12 -m venv venv
source venv/bin/activate
uv pip install --upgrade pip
uv pip install -r requirements.txt
echo "✅ Python environment setup complete"
echo ""

# Step 2: Compile the pipeline
echo "Step 2: Compiling LakeFS initial data pipeline..."
python create_initial_data_pipeline.py --output lakefs_initial_data_pipeline.yaml
echo "✅ Pipeline compiled successfully"
echo ""

# Step 3: Submit to Kubeflow
echo "Step 3: Submitting pipeline to Kubeflow..."
python -c "
import os
import sys
from utils.kubeflow.pipeline_submitter import submit_pipeline

result = submit_pipeline(
    pipeline_path='lakefs_initial_data_pipeline.yaml',
    pipeline_name='lakefs-initial-data-setup',
    experiment_name='initial-data-setup',
    run_name=f'initial-data-setup-manual',
    additional_params={'lakefs_repository': 'mlops-data'}
)

print('Pipeline submitted successfully!')
print(f'Pipeline ID: {result[\"pipeline_id\"]}')
print(f'Experiment ID: {result[\"experiment_id\"]}')
print(f'Run ID: {result[\"run_id\"]}')
print(f'Git Commit: {result[\"git_commit\"][:8]} if result[\"git_commit\"] != \"unknown\" else \"unknown\"')
"
echo "✅ Pipeline submitted to Kubeflow"
echo ""

# Step 4: Summary
echo "========================================="
echo "✅ Initial Data Setup Completed"
echo "========================================="
echo ""
echo "What was accomplished:"
echo "  1. ✅ Compiled LakeFS initial data pipeline"
echo "  2. ✅ Submitted pipeline to Kubeflow"
echo ""
echo "The pipeline will:"
echo "  1. Create LakeFS repository 'mlops-data' if it doesn't exist"
echo "  2. Initialize main branch"
echo "  3. Upload iris dataset to main branch"
echo "  4. Create dataset metadata"
echo "  5. Add README documentation"
echo ""
echo "Monitor the pipeline:"
echo "  - Kubeflow UI: http://localhost:8080"
echo "  - LakeFS UI: http://localhost:8000"
echo ""
echo "After completion, the iris dataset will be available at:"
echo "  s3://mlops-data/main/iris/dataset.json"
echo ""
echo "Future pipelines can now use versioned data from LakeFS!"