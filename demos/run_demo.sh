#!/bin/bash
# ML Pipeline Demo Runner
# Quick setup and execution script for demos

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎯 ML Pipeline Framework Demo Runner${NC}"
echo "=================================="

# Function to check if MLflow is running
check_mlflow() {
    if ! curl -s http://localhost:5000 > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠ MLflow server not running. Starting it up...${NC}"
        echo "Starting MLflow server in background..."
        nohup mlflow server --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
        sleep 5
        if curl -s http://localhost:5000 > /dev/null 2>&1; then
            echo -e "${GREEN}✓ MLflow server started successfully${NC}"
        else
            echo -e "${RED}✗ Failed to start MLflow server. Check mlflow.log${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ MLflow server is running${NC}"
    fi
}

# Function to show dataset info
show_dataset_info() {
    echo -e "\n${BLUE}📊 Available Datasets:${NC}"
    echo "====================="
    
    if [ -f "data/telco_customer_churn.csv" ]; then
        rows=$(wc -l < data/telco_customer_churn.csv)
        echo -e "${GREEN}1. Telco Customer Churn${NC} - $rows rows (Classification)"
    fi
    
    if [ -f "data/wine_quality_red.csv" ]; then
        rows=$(wc -l < data/wine_quality_red.csv)
        echo -e "${GREEN}2. Wine Quality${NC} - $rows rows (Regression)"
    fi
    
    if [ -f "data/titanic.csv" ]; then
        rows=$(wc -l < data/titanic.csv)
        echo -e "${GREEN}3. Titanic Survival${NC} - $rows rows (Classification)"
    fi
    
    if [ -f "data/adult_income.csv" ]; then
        rows=$(wc -l < data/adult_income.csv)
        echo -e "${GREEN}4. Adult Income${NC} - $rows rows (Classification)"
    fi
}

# Main menu
show_menu() {
    echo -e "\n${BLUE}🎯 Choose a demo to run:${NC}"
    echo "========================"
    echo "1. Telco Customer Churn (Classification) - ~20 min"
    echo "2. Wine Quality (Regression) - ~15 min"
    echo "3. Titanic Survival (Classification) - ~10 min"
    echo "4. Adult Income (Classification) - ~40 min"
    echo ""
    echo "v. Validate all configs"
    echo "s. Show system status"
    echo "x. Exit"
    echo ""
}

# Change to demos directory if not already there
if [ ! -d "data" ] && [ -d "demos/data" ]; then
    cd demos
    echo -e "${YELLOW}📁 Changed to demos directory${NC}"
fi

# Check if we're in the right directory
if [ ! -d "data" ] || [ ! -d "configs" ]; then
    echo -e "${RED}✗ Error: Must be run from demos directory or project root${NC}"
    echo "Current directory: $(pwd)"
    echo "Please navigate to the demos directory or project root"
    exit 1
fi

# Check prerequisites
echo -e "\n${YELLOW}🔧 Checking prerequisites...${NC}"

# Check if mlpipeline is installed
if ! command -v mlpipeline &> /dev/null; then
    echo -e "${RED}✗ mlpipeline command not found${NC}"
    echo "Please install the ML Pipeline framework:"
    echo "cd /path/to/mlpipeline && pip install -e ."
    exit 1
fi

echo -e "${GREEN}✓ mlpipeline is installed${NC}"

# Check MLflow
check_mlflow

# Show datasets
show_dataset_info

# Main loop
while true; do
    show_menu
    read -p "Enter your choice: " choice
    
    case $choice in
        1)
            echo -e "\n${BLUE}🚀 Running Telco Churn demo...${NC}"
            mlpipeline train --config configs/telco_churn_demo.yaml --experiment-id "demo_telco_$(date +%Y%m%d_%H%M%S)"
            ;;
        2)
            echo -e "\n${BLUE}🚀 Running Wine Quality demo...${NC}"
            mlpipeline train --config configs/wine_quality_demo.yaml --experiment-id "demo_wine_$(date +%Y%m%d_%H%M%S)"
            ;;
        3)
            echo -e "\n${BLUE}🚀 Running Titanic demo...${NC}"
            mlpipeline train --config configs/titanic_survival_demo.yaml --experiment-id "demo_titanic_$(date +%Y%m%d_%H%M%S)"
            ;;
        4)
            echo -e "\n${BLUE}🚀 Running Adult Income demo...${NC}"
            mlpipeline train --config configs/adult_income_demo.yaml --experiment-id "demo_income_$(date +%Y%m%d_%H%M%S)"
            ;;
        v)
            echo -e "\n${YELLOW}🔍 Validating all configurations...${NC}"
            mlpipeline validate --config configs/telco_churn_demo.yaml
            mlpipeline validate --config configs/wine_quality_demo.yaml
            mlpipeline validate --config configs/titanic_survival_demo.yaml
            mlpipeline validate --config configs/adult_income_demo.yaml
            ;;
        s)
            echo -e "\n${YELLOW}🔧 System Status:${NC}"
            mlpipeline status --detailed
            ;;
        x)
            echo -e "${GREEN}👋 Thanks for trying ML Pipeline demos!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            ;;
    esac
    
    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read
done
