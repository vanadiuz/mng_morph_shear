from abc import ABC, abstractmethod
class Particle(ABC):

	def __init__(self):
		print("A particle base has been created!")
    

	@abstractmethod
	def load(self):
		#loads coordinates and other staff
		pass

	@abstractmethod
	def initialize(self, system, parameters):
		#creates particles in simulation box with bonded int. + WCA
		pass
