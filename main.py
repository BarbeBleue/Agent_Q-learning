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

<<<<<<< HEAD
	i = 0
	while i < nb_exp:
		sim.experiment_run()
		i += 1

#sim.generate_maps(301,50) #to generate maps
=======
i=0
while(i<1):
	#sim.load_quicksave()

	#sim.env.agent.brain._lr = 0.01 + i*0.01

	sim.experiment_run(replay=False, delay = 0.0, nomap=True,nb_test=50)
	i += 1	
>>>>>>> 53b868524ca6f4b445b7806d5328c92d62e6905b
