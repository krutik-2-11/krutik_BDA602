#!/bin/bash

# create the output folder in the root directory of the container
mkdir /output

# create the output.txt file inside the output folder
touch /output/output.csv


# pause the script for 10 seconds
sleep 10

echo "Entered the main shell. Creating Database..."

# create database
mariadb -u root -h db_container -p1998 -e "CREATE DATABASE IF NOT EXISTS baseball;"

echo "Database baseball created!"

# import SQL file into database
mariadb -u root -h db_container -p1998 baseball < baseball.sql

echo "baseball data loaded in the database!"

mariadb -u root -h db_container -p1998 baseball < sample_script.sql

echo "Rolling batting average query ran successfully!"

mariadb -u root -h db_container -p1998 -e "select * from baseball.fl_rolling_batting_average" > /output/output.csv

echo "Output file output.csv generated!"




