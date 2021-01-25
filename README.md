# Reinforcement Learning Project Game

## Project structure
The main file is "agent.py" including the algorithm/agent etc. To play the game by yourself, have a look in "game for humans" folder,
the game adapted to the agaent is located inside "game for computer" folder. The agent is stored inside the stored_agent folder. 
The file "significance_test.py" only includes a small script for evaluation purposes.

## Installation guide
Very important: I adapted the game file "car_racing.py" and called the adapted version "car_racing_custom.py" since I needed to create some pivotal changes inside this file.
This file is inside the git repository located in "game for computer", but must be copied to the python installation library which is e. g. here:
C:\Users\NAME_OF_USER\AppData\Local\Programs\Python\Python37\Lib\site-packages\gym\envs\box2d
In the folder there is a file called "__init__.py". Also this files needs to replaced by a new version put into "game for computer" folder.

## Run guide
Just compile the file "agent.py". In order to change the settings such as the algorithms and its parameters, change it inside the "agent.py"-file. 
