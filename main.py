"""Run to play the simulation of an agent in a maze full of obstacles, enemies and food"""

import time
from simulator import *
import cProfile


sim = Simulator()

print("\nTo use the simulator in manual mode press 'M' or 'm'.")
print("To run the experiment described in [Lin1992] press any other touch.")
print("To run the simulator with the best recorded netwotk press '*'.")
c_input=input()

if c_input == "M" or c_input == "m":
	sim.manual_run()

elif c_input =='*':
	sim.saved_experiment_run()

else:
	#change value to adjust the number of independent experiments to run
	nb_exp = 1
	i = 0
	while i < nb_exp:
		sim.experiment_run()
		i += 1
