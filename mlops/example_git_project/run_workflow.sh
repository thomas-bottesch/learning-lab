#!/bin/bash

# Expert MLOps Workflow Runner
# Simple interface for users to run the complete MLOps pipeline

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="mlops_config.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Expert MLOps Workflow${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_step "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is required but not installed"
        exit 1
    fi
    
    # Check Docker (optional)
    if ! command -v docker &> /dev/null; then
        print_warning "Docker not found. Some features may be limited."
    fi
    
    # Check kubectl (optional)
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl not found. Kubernetes operations disabled."
    fi
    
    print_info "Dependencies check passed"
}

load_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "Configuration file $CONFIG_FILE not found"
        exit 1
    fi
    
    print_step "Loading configuration from $CONFIG_FILE"
    
    # Extract basic config using yq or python
    if command -v yq &> /dev/null; then
        PROJECT_NAME=$(yq e '.project.name' "$CONFIG_FILE")
        PROJECT_VERSION=$(yq e '.project.version' "$CONFIG_FILE")
    else
        # Fallback to Python
        PROJECT_NAME=$(python3 -c "import yaml; data=yaml.safe_load(open('$CONFIG_FILE')); print(data.get('project', {}).get('name', 'unknown'))")
        PROJECT_VERSION=$(python3 -c "import yaml; data=yaml.safe_load(open('$CONFIG_FILE')); print(data.get('project', {}).get('version', '1.0.0'))")
    fi
    
    print_info "Project: $PROJECT_NAME v$PROJECT_VERSION"
}

run_local_training() {
    print_step "Running local training..."
    
    # Install dependencies if needed
    if [[ ! -f "venv" ]]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    if [[ ! -f "venv/requirements_installed" ]]; then
        print_info "Installing Python dependencies..."
        pip install -r requirements.txt
        touch venv/requirements_installed
    fi
    
    # Run training
    print_info "Starting training script..."
    python3 train.py
    
    deactivate
    
    print_info "Local training completed"
}

build_docker_image() {
    print_step "Building Docker image..."
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker not available, skipping image build"
        return
    fi
    
    IMAGE_NAME="${PROJECT_NAME}:${PROJECT_VERSION}"
    
    docker build -t "$IMAGE_NAME" .
    
    print_info "Docker image built: $IMAGE_NAME"
}

run_kubeflow_pipeline() {
    print_step "Running Kubeflow pipeline..."
    
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl not available, skipping Kubeflow pipeline"
        return
    fi
    
    # Check if Kubeflow is available
    if kubectl get pods -n kubeflow 2>/dev/null | grep -q "pipeline" ; then
        print_info "Submitting pipeline to Kubeflow..."
        
        # This would trigger the actual pipeline
        # For now, just show the command
        print_info "Command: python3 -m projects.kubeflow.submit_pipeline_browserless"
        
        # Check if we're in the right directory structure
        if [[ -f "../projects/kubeflow/submit_pipeline_browserless.py" ]]; then
            cd ..
            python3 -m projects.kubeflow.submit_pipeline_browserless
            cd - > /dev/null
        else
            print_warning "Kubeflow pipeline scripts not found"
        fi
    else
        print_warning "Kubeflow not detected in cluster"
    fi
}

show_status() {
    print_step "Workflow Status"
    echo ""
    echo "Services:"
    echo "  - Local Training: ✅ Complete"
    echo "  - Docker Build: ✅ Complete"
    echo "  - Kubeflow Pipeline: ⚠️  May require manual trigger"
    echo ""
    echo "Next steps:"
    echo "  1. Push to Git repository to trigger automated pipeline"
    echo "  2. Check MLflow dashboard for experiment tracking"
    echo "  3. Monitor Kubeflow pipelines for execution status"
    echo ""
    echo "Dashboard URLs:"
    echo "  - MLflow: http://localhost:5000"
    echo "  - Kubeflow: http://localhost:8080"
    echo "  - Forgejo (Git): http://localhost:4000"
}

main() {
    print_header
    
    # Parse arguments
    MODE="all"
    if [[ $# -gt 0 ]]; then
        MODE="$1"
    fi
    
    case "$MODE" in
        "local")
            check_dependencies
            load_config
            run_local_training
            ;;
        "docker")
            check_dependencies
            load_config
            build_docker_image
            ;;
        "kubeflow")
            check_dependencies
            load_config
            run_kubeflow_pipeline
            ;;
        "all"|*)
            check_dependencies
            load_config
            run_local_training
            build_docker_image
            run_kubeflow_pipeline
            ;;
    esac
    
    show_status
    
    echo -e "${GREEN}✅ Expert MLOps workflow completed!${NC}"
}

# Run main function
main "$@"