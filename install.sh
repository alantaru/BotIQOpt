#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
  echo "Creating .env file"
  touch .env
fi

# Create conf file if it doesn't exist
if [ ! -f "conf.ini" ]; then
  echo "Creating conf file"
  touch conf.ini
fi

echo "Installation complete!"
