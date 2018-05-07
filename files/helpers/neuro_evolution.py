import numpy as np

class MarioNeuroEvolution():
	def __init__(self, button_names):

		self.BOX_RADIUS = 6
		self.INPUT_SIZE = (self.BOX_RADIUS * 2 + 1) * (self.BOX_RADIUS * 2 + 1)

		self.INPUTS = self.INPUT_SIZE + 1
		self.OUTPUTS = len(button_names)

		self.POPULATION = 300
		self.DELTA_DISJOINT = 2.0
		self.DELTA_WEIGHTS = 0.4
		self.DELTA_THRESHOLD = 1.0

		self.STALE_SPECIES = 15

		self.MUTATE_CONNECTIONS_CHANCE = 0.25
		self.PERTURB_CHANCE = 0.90
		self.CROSSOVER_CHANCE = 0.75
		self.LINK_MUTATION_CHANCE = 2.0
		self.NODE_MUTATION_CHANCE = 0.50
		self.BIAS_MUTATION_CHANCE = 0.40
		self.STEP_SIZE = 0.1
		self.DISABLE_MUTATION_CHANCE = 0.4
		self.ENABLE_MUTATION_CHANCE = 0.2

		self.TIMEOUT_CONSTANT = 20

		self.MAX_NODES = 1000000

		self.initializePool()

	def initializePool(self):
		self.pool = self.newPool()

		for i in range(0, self.POPULATION):
			basic = self.basicGenome()
			self.addToSpecies(basic)

		self.initializeRun()

	def newPool(self): # 1
		pool = dict()
		pool["species"] = []
		pool["generation"] = 0
		pool["innovation"] = self.OUTPUTS
		pool["currentSpecies"] = 1
		pool["currentGenome"] = 1
		pool["currentFrame"] = 0
		pool["maxFitness"] = 0
	
		return pool

	def basicGenome(self): # 2
		genome = self.newGenome()
		innovation = 1

		genome.maxneuron = self.INPUTS
		self.mutate(genome)
	
		return genome

	def newGenome(self): # 2.1
		genome = dict()
		genome["genes"] = []
		genome["fitness"] = 0
		genome["adjustedFitness"] = 0
		genome["network"] = dict()
		genome["maxneuron"] = 0
		genome["globalRank"] = 0
		genome["mutationRates"] = dict()
		genome["mutationRates"]["connections"] = self.MUTATE_CONNECTIONS_CHANCE
		genome["mutationRates"]["link"] = self.LINK_MUTATION_CHANCE
		genome["mutationRates"]["bias"] = self.BIAS_MUTATION_CHANCE
		genome["mutationRates"]["node"] = self.NODE_MUTATION_CHANCE
		genome["mutationRates"]["enable"] = self.ENABLE_MUTATION_CHANCE
		genome["mutationRates"]["disable"] = self.DISABLE_MUTATION_CHANCE
		genome["mutationRates"]["step"] = self.STEP_SIZE
	
		return genome

	def mutate(self, genome): # 2.2
		for mutation,rate in genome["mutationRates"].items():
			if (np.random.randint(1,3) == 1):
				genome["mutationRates"][mutation] = 0.95 * rate
			else
				genome["mutationRates"][mutation] = 1.05263 * rate

		if (np.random.rand() < genome["mutationRates"]["connections"]):
			self.pointMutate(genome)
	
		p = genome["mutationRates"]["link"]
		while (p > 0):
			if (np.random.rand() < p):
				self.linkMutate(genome, false)
			p = p - 1

		p = genome["mutationRates"]["bias"]
		while (p > 0):
			if (np.random.rand() < p):
				self.linkMutate(genome, true)
			p = p - 1
		
		p = genome["mutationRates"]["node"]
		while (p > 0):
			if (np.random.rand() < p):
				self.nodeMutate(genome)
			p = p - 1
		
		p = genome["mutationRates"]["enable"]
		while (p > 0):
			if (np.random.rand() < p):
				self.enableDisableMutate(genome, true)
			p = p - 1

		p = genome["mutationRates"]["disable"]
		while (p > 0):
			if np.random.rand() < p):
				self.enableDisableMutate(genome, false)
			p = p - 1

	def pointMutate(self, ???): #2.2.1
		pass

	def linkMutate(self, ???, ???): #2.2.2
		pass

	def nodeMutate(self, ???): #2.2.3
		pass

	def enableDisableMutate(self, ???, ???): #2.2.4
		pass

	def addToSpecies(self, ???): # 3
		pass

	def initializeRun(self): # 4
		pass