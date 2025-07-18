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

# Function to validate demo
validate_demo() {
    local config_file=$1
    local demo_name=$2
    
    echo -e "\n${YELLOW}🔍 Validating $demo_name demo...${NC}"
    
    if mlpipeline validate --config "$config_file"; then
        echo -e "${GREEN}✓ Configuration is valid${NC}"
        return 0
    else
        echo -e "${RED}✗ Configuration validation failed${NC}"
        return 1
    fi
}

# Function to run quick demo
run_quick_demo() {
    local config_file=$1
    local demo_name=$2
    local experiment_id="demo_$(date +%Y%m%d_%H%M%S)"
    
    echo -e "\n${BLUE}🚀 Running $demo_name demo...${NC}"
    echo "Experiment ID: $experiment_id"
    
    # Create a temporary config with reduced trials for quick demo
    local temp_config="/tmp/quick_${demo_name}.yaml"
    
    # Copy original config and modify for quick run
    cp "$config_file" "$temp_config"
    
    # Reduce trials for quick demo (using sed to replace)
    sed -i.bak 's/n_trials: [0-9]*/n_trials: 5/' "$temp_config"
    sed -i.bak 's/timeout: [0-9]*/timeout: 300/' "$temp_config"
    
    echo -e "${YELLOW}⚡ Quick demo mode: 5 trials, 5-minute timeout${NC}"
    
    if mlpipeline train --config "$temp_config" --experiment-id "$experiment_id"; then
        echo -e "\n${GREEN}🎉 Demo completed successfully!${NC}"
        echo -e "${BLUE}📊 View results at: http://localhost:5000${NC}"
        echo -e "${BLUE}📝 Logs: results/${demo_name}_demo.log${NC}"
        
        # Show experiment results
        echo -e "\n${YELLOW}📈 Latest Experiments:${NC}"
        mlpipeline experiments --limit 3 --sort-by accuracy || echo "No experiments found"
        
        rm -f "$temp_config" "$temp_config.bak"
        return 0
    else
        echo -e "${RED}✗ Demo failed${NC}"
        rm -f "$temp_config" "$temp_config.bak"
        return 1
    fi
}

# Function to run full demo
run_full_demo() {
    local config_file=$1
    local demo_name=$2
    local experiment_id="full_demo_$(date +%Y%m%d_%H%M%S)"
    
    echo -e "\n${BLUE}🚀 Running FULL $demo_name demo...${NC}"
    echo "Experiment ID: $experiment_id"
    echo -e "${YELLOW}⏰ This may take 15-60 minutes depending on the dataset${NC}"
    
    if mlpipeline train --config "$config_file" --experiment-id "$experiment_id"; then
        echo -e "\n${GREEN}🎉 Full demo completed successfully!${NC}"
        echo -e "${BLUE}📊 View results at: http://localhost:5000${NC}"
        echo -e "${BLUE}📝 Logs: results/${demo_name}_demo.log${NC}"
        
        # Show experiment results
        echo -e "\n${YELLOW}📈 Latest Experiments:${NC}"
        mlpipeline experiments --limit 5 --sort-by accuracy || echo "No experiments found"
        
        return 0
    else
        echo -e "${RED}✗ Full demo failed${NC}"
        return 1
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
    echo "Run modes:"
    echo "q. Quick demo (5 trials, 5 min timeout)"
    echo "f. Full demo (full hyperparameter optimization)"
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
            read -p "Quick demo (q) or Full demo (f)? " mode
            if validate_demo "configs/telco_churn_demo.yaml" "Telco Churn"; then
                if [ "$mode" = "q" ]; then
                    run_quick_demo "configs/telco_churn_demo.yaml" "telco_churn"
                else
                    run_full_demo "configs/telco_churn_demo.yaml" "telco_churn"
                fi
            fi
            ;;
        2)
            read -p "Quick demo (q) or Full demo (f)? " mode
            if validate_demo "configs/wine_quality_demo.yaml" "Wine Quality"; then
                if [ "$mode" = "q" ]; then
                    run_quick_demo "configs/wine_quality_demo.yaml" "wine_quality"
                else
                    run_full_demo "configs/wine_quality_demo.yaml" "wine_quality"
                fi
            fi
            ;;
        3)
            read -p "Quick demo (q) or Full demo (f)? " mode
            if validate_demo "configs/titanic_survival_demo.yaml" "Titanic"; then
                if [ "$mode" = "q" ]; then
                    run_quick_demo "configs/titanic_survival_demo.yaml" "titanic_survival"
                else
                    run_full_demo "configs/titanic_survival_demo.yaml" "titanic_survival"
                fi
            fi
            ;;
        4)
            read -p "Quick demo (q) or Full demo (f)? " mode
            if validate_demo "configs/adult_income_demo.yaml" "Adult Income"; then
                if [ "$mode" = "q" ]; then
                    run_quick_demo "configs/adult_income_demo.yaml" "adult_income"
                else
                    run_full_demo "configs/adult_income_demo.yaml" "adult_income"
                fi
            fi
            ;;
        q)
            echo -e "\n${YELLOW}🏃‍♂️ Quick Demo Mode${NC}"
            echo "1. Telco Churn"
            echo "2. Wine Quality" 
            echo "3. Titanic"
            echo "4. Adult Income"
            read -p "Which demo? (1-4): " demo_choice
            
            case $demo_choice in
                1) run_quick_demo "configs/telco_churn_demo.yaml" "telco_churn" ;;
                2) run_quick_demo "configs/wine_quality_demo.yaml" "wine_quality" ;;
                3) run_quick_demo "configs/titanic_survival_demo.yaml" "titanic_survival" ;;
                4) run_quick_demo "configs/adult_income_demo.yaml" "adult_income" ;;
                *) echo -e "${RED}Invalid choice${NC}" ;;
            esac
            ;;
        f)
            echo -e "\n${YELLOW}🎯 Full Demo Mode${NC}"
            echo "1. Telco Churn"
            echo "2. Wine Quality"
            echo "3. Titanic" 
            echo "4. Adult Income"
            read -p "Which demo? (1-4): " demo_choice
            
            case $demo_choice in
                1) run_full_demo "configs/telco_churn_demo.yaml" "telco_churn" ;;
                2) run_full_demo "configs/wine_quality_demo.yaml" "wine_quality" ;;
                3) run_full_demo "configs/titanic_survival_demo.yaml" "titanic_survival" ;;
                4) run_full_demo "configs/adult_income_demo.yaml" "adult_income" ;;
                *) echo -e "${RED}Invalid choice${NC}" ;;
            esac
            ;;
        v)
            echo -e "\n${YELLOW}🔍 Validating all configurations...${NC}"
            validate_demo "configs/telco_churn_demo.yaml" "Telco Churn"
            validate_demo "configs/wine_quality_demo.yaml" "Wine Quality"
            validate_demo "configs/titanic_survival_demo.yaml" "Titanic"
            validate_demo "configs/adult_income_demo.yaml" "Adult Income"
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