#!/bin/bash

# pause the script for 30 seconds

echo "Entered the main shell. Taking rest for 30 sec....."

sleep 30

if mariadb -u root -p1998 -h db_container -e "select * from baseball.batter_counts limit 1;"
then
    echo "Database baseball exists, adding my features into baseball database..."
    mariadb -u root -p1998 -h db_container baseball < Krutik_Baseball_Features_Final.sql
    echo "Features added to the baseball database!"
else
    echo "Database baseball does not exist. Creating the database baseball..."
    mariadb -u root -p1998 -h db_container baseball < baseball.sql
    echo "Database baseball created and loaded successfully! Adding the features..."
    mariadb -u root -p1998 -h db_container baseball < Krutik_Baseball_Features_Final.sql
    echo "Features added to baseball database!"
fi


echo "Running the Python Script to generate final report..."

python3 main.py

echo "Script run successfully! Please check the html file for final report."







