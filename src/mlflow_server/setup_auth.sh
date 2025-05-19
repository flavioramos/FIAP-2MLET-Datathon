#!/bin/bash

# Install apache2-utils for htpasswd command
apt-get update && apt-get install -y apache2-utils

# Create .htpasswd file with a user (you can change the username and password)
htpasswd -cb .htpasswd teacher mlet-password

# Create the auth directory if it doesn't exist
mkdir -p auth
mv .htpasswd auth/

echo "Authentication setup complete. Username: teacher, Password: mlet-password" 