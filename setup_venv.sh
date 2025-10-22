#!/bin/bash

# Simple setup script for Ultralytics YOLO environment
# Usage: ./setup_venv_simple.sh

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Setting up Ultralytics Environment${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Get workspace directory
WORKSPACE_DIR="$(pwd)"
VENV_DIR="$WORKSPACE_DIR/venv_tamt"

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
else
    echo "Creating virtual environment..."
    python3 -m venv venv_tamt
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
source venv_tamt/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Ultralytics (this will install all dependencies including PyTorch, OpenCV, etc.)
echo "Installing Ultralytics..."
pip install ultralytics

echo -e "${GREEN}✓ Ultralytics installed${NC}"

# Test import
echo "Testing installation..."
python3 -c "from ultralytics import YOLO; print('✓ Ultralytics working!')"

# Create simple activation script
cat > setup_env.sh << 'EOF'
#!/bin/bash
# Activate environment
source install/setup.bash 2>/dev/null || true
source venv_tamt/bin/activate
echo "✓ Environment activated"
EOF

chmod +x setup_env.sh

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "To activate in the future, run:"
echo -e "  ${BLUE}source setup_env.sh${NC}"
echo ""