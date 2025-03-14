#!/bin/bash

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info >= (3, 9))')
if [ "$PYTHON_VERSION" != "True" ]; then
  echo "Python 3.9 or higher is required"
  exit 1
fi

# Install system dependencies for TA-Lib
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev
sudo apt-get install -y libjpeg-dev zlib1g-dev libfreetype6-dev
sudo apt-get install -y libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt-get install -y libxml2-dev libxslt1-dev libpq-dev
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Install TA-Lib
echo "Installing TA-Lib..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib/ ta-lib-0.4.0-src.tar.gz

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify TA-Lib installation
echo "Verifying TA-Lib installation..."
python3 -c "import talib; print('TA-Lib version:', talib.__version__)" || {
  echo "TA-Lib installation failed!"
  exit 1
}

# Create configuration files
echo "Creating configuration files..."
[ ! -f ".env" ] && touch .env
[ ! -f "conf.ini" ] && touch conf.ini

echo -e "\nInstallation complete!"
echo -e "\nTo activate the virtual environment, run:"
echo "source .venv/bin/activate"
echo -e "\nTo deactivate, run:"
echo "deactivate"
