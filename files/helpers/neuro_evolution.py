import numpy as np
import json
import math
from xml.dom import minidom


class NEAT():
	def __init__(self, output_size):
		try:
			doc = minidom.parse('plugins/SerpentDonkeyKongGameAgentPlugin/files/helpers/neat_config.xml')

		except FileNotFoundError:
			self._create_config_file()
			doc = minidom.parse('plugins/SerpentDonkeyKongGameAgentPlugin/files/helpers/neat_config.xml')

		self.HEIGHT_RADIUS = int(doc.getElementsByTagName("height_radius")[0].firstChild.nodeValue.strip()) #2
		self.WIDTH_RADIUS = int(doc.getElementsByTagName("height_radius")[0].firstChild.nodeValue.strip()) #2
		self.INPUT_SIZE = (self.HEIGHT_RADIUS + 1) * (self.WIDTH_RADIUS * 2 + 1) + 1

		self.INPUTS = self.INPUT_SIZE + 1
		self.OUTPUTS = output_size

		self.POPULATION = int(doc.getElementsByTagName("population")[0].firstChild.nodeValue.strip()) #5
		self.DELTA_DISJOINT = float(doc.getElementsByTagName("delta_disjoint")[0].firstChild.nodeValue.strip()) #2.0
		self.DELTA_WEIGHTS = float(doc.getElementsByTagName("delta_weight")[0].firstChild.nodeValue.strip()) #0.4
		self.DELTA_THRESHOLD = float(doc.getElementsByTagName("delta_threshold")[0].firstChild.nodeValue.strip()) #1.0

		self.STALE_SPECIES = int(doc.getElementsByTagName("stale_species")[0].firstChild.nodeValue.strip()) #15

		self.MUTATE_CONNECTIONS_CHANCE = float(doc.getElementsByTagName("mutate_connections")[0].firstChild.nodeValue.strip()) #0.25
		self.PERTURB_CHANCE = float(doc.getElementsByTagName("perturb")[0].firstChild.nodeValue.strip()) #0.90
		self.CROSSOVER_CHANCE = float(doc.getElementsByTagName("crossover")[0].firstChild.nodeValue.strip()) #0.75
		self.LINK_MUTATION_CHANCE = float(doc.getElementsByTagName("link_mutation")[0].firstChild.nodeValue.strip()) #2.0
		self.NODE_MUTATION_CHANCE = float(doc.getElementsByTagName("node_mutation")[0].firstChild.nodeValue.strip()) #0.50
		self.BIAS_MUTATION_CHANCE = float(doc.getElementsByTagName("bias_mutation")[0].firstChild.nodeValue.strip()) #0.40
		self.STEP_SIZE = float(doc.getElementsByTagName("step_size")[0].firstChild.nodeValue.strip()) #0.1
		self.DISABLE_MUTATION_CHANCE = float(doc.getElementsByTagName("disable_mutation")[0].firstChild.nodeValue.strip()) #0.4
		self.ENABLE_MUTATION_CHANCE = float(doc.getElementsByTagName("enable_mutation")[0].firstChild.nodeValue.strip()) #0.2

		self.TIMEOUT_CONSTANT = int(doc.getElementsByTagName("timeout_constant")[0].firstChild.nodeValue.strip()) #20

		self.MAX_NODES = int(doc.getElementsByTagName("max_nodes")[0].firstChild.nodeValue.strip()) #1000000

		start_with_gen = int(doc.getElementsByTagName("start_with_gen")[0].firstChild.nodeValue.strip())

		if (start_with_gen == -1):
			self.generation_evaluated = False
			self._initialize_pool()
		else :
			self.generation_evaluated = True
			self._load_generation(start_with_gen)

		print("Gen " + str(self.pool["generation"]) + " - Specie " + str(self.pool["currentSpecies"]) + " - Ind " + str(self.pool["currentGenome"]) + " - Score : ", end='', flush=True)

	def _load_generation(self, start_with_gen):
		with open('plugins/SerpentDonkeyKongGameAgentPlugin/files/helpers/neat_gen_' + str(start_with_gen) + '.json') as f:
			self.pool = json.load(f)

	def _create_config_file(self):
		print("No config file found for Neuro-Evolution.\n Creation of the default config file.")
		doc = minidom.Document()

		config = doc.createElement('config')

		inputs = doc.createElement('inputs')
		height_radius = doc.createElement('height_radius')
		height_radius.appendChild(doc.createTextNode("2"))
		width_radius = doc.createElement('width_radius')
		width_radius.appendChild(doc.createTextNode("2"))
		inputs.appendChild(height_radius)
		inputs.appendChild(width_radius)

		population = doc.createElement('population')
		population.appendChild(doc.createTextNode("300"))

		deltas = doc.createElement('deltas')
		delta_disjoint = doc.createElement('delta_disjoint')
		delta_disjoint.appendChild(doc.createTextNode("2.0"))
		delta_weight = doc.createElement('delta_weight')
		delta_weight.appendChild(doc.createTextNode("0.4"))
		delta_threshold = doc.createElement('delta_threshold')
		delta_threshold.appendChild(doc.createTextNode("1.0"))
		deltas.appendChild(delta_disjoint)
		deltas.appendChild(delta_weight)
		deltas.appendChild(delta_threshold)

		stale_species = doc.createElement("stale_species")
		stale_species.appendChild(doc.createTextNode("15"))

		chances = doc.createElement("chances")
		mutate_connections = doc.createElement("mutate_connections")
		mutate_connections.appendChild(doc.createTextNode("0.25"))
		perturb = doc.createElement("perturb")
		perturb.appendChild(doc.createTextNode("0.90"))
		crossover = doc.createElement("crossover")
		crossover.appendChild(doc.createTextNode("0.75"))
		link_mutation = doc.createElement("link_mutation")
		link_mutation.appendChild(doc.createTextNode("2.0"))
		node_mutation = doc.createElement("node_mutation")
		node_mutation.appendChild(doc.createTextNode("0.50"))
		bias_mutation = doc.createElement("bias_mutation")
		bias_mutation.appendChild(doc.createTextNode("0.40"))
		step_size = doc.createElement("step_size")
		step_size.appendChild(doc.createTextNode("0.1"))
		disable_mutation = doc.createElement("disable_mutation")
		disable_mutation.appendChild(doc.createTextNode("0.4"))
		enable_mutation = doc.createElement("enable_mutation")
		enable_mutation.appendChild(doc.createTextNode("0.2"))
		chances.appendChild(mutate_connections)
		chances.appendChild(perturb)
		chances.appendChild(crossover)
		chances.appendChild(link_mutation)
		chances.appendChild(node_mutation)
		chances.appendChild(bias_mutation)
		chances.appendChild(step_size)
		chances.appendChild(disable_mutation)
		chances.appendChild(enable_mutation)

		timeout_constant = doc.createElement("timeout_constant")
		timeout_constant.appendChild(doc.createTextNode("20"))

		max_nodes = doc.createElement("max_nodes")
		max_nodes.appendChild(doc.createTextNode("1000000"))

		start_with_gen = doc.createElement("start_with_gen")
		start_with_gen.appendChild(doc.createTextNode("-1"))

		config.appendChild(inputs)
		config.appendChild(population)
		config.appendChild(deltas)
		config.appendChild(stale_species)
		config.appendChild(chances)
		config.appendChild(timeout_constant)
		config.appendChild(max_nodes)
		config.appendChild(start_with_gen)

		doc.appendChild(config)

		doc.writexml(open('plugins/SerpentDonkeyKongGameAgentPlugin/files/helpers/neat_config.xml', 'w'), indent="    ", addindent="  ", newl='\n')


	def _initialize_pool(self):
		self._new_pool()

		for i in range(self.POPULATION):
			basic = self._basic_genome()
			self._add_to_species(basic)

	def _new_pool(self):
		self.pool = dict()
		self.pool["species"] = []
		self.pool["generation"] = 0
		self.pool["innovation"] = self.OUTPUTS
		self.pool["currentSpecies"] = 0
		self.pool["currentGenome"] = 0
		self.pool["currentFrame"] = 0
		self.pool["maxFitness"] = 0

	def _basic_genome(self):
		genome = self._new_genome()
		innovation = 1

		genome["maxneuron"] = self.INPUTS
		self._mutate(genome)
	
		return genome

	def _new_genome(self):
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

	def _mutate(self, genome):
		for mutation,rate in genome["mutationRates"].items():
			if (np.random.randint(1,3) == 1):
				genome["mutationRates"][mutation] = 0.95 * rate
			else:
				genome["mutationRates"][mutation] = 1.05263 * rate

		if (np.random.rand() < genome["mutationRates"]["connections"]):
			self._point_mutate(genome)
	
		p = genome["mutationRates"]["link"]
		while (p > 0):
			if (np.random.rand() < p):
				self._link_mutate(genome, False)
			p = p - 1

		p = genome["mutationRates"]["bias"]
		while (p > 0):
			if (np.random.rand() < p):
				self._link_mutate(genome, True)
			p = p - 1
	
		p = genome["mutationRates"]["node"]
		while (p > 0):
			if (np.random.rand() < p):
				self._node_mutate(genome)
			p = p - 1
	
		p = genome["mutationRates"]["enable"]
		while (p > 0):
			if (np.random.rand() < p):
				self._enable_disable_mutate(genome, True)
			p = p - 1

		p = genome["mutationRates"]["disable"]
		while (p > 0):
			if (np.random.rand() < p):
				self._enable_disable_mutate(genome, False)
			p = p - 1

	def _enable_disable_mutate(self, genome, enable):
		candidates = []
		for gene in genome["genes"]:
			if (gene["enabled"] == (not enable)):
				candidates.append(gene)
		
		if (len(candidates) == 0):
			return
	
		gene = candidates[np.random.randint(0,len(candidates))]
		gene["enabled"] = not gene["enabled"]

	def _node_mutate(self, genome):
		if (len(genome["genes"]) == 0):
			return

		genome["maxneuron"] = genome["maxneuron"] + 1

		gene = genome["genes"][np.random.randint(0,len(genome["genes"]))]
		if (not gene["enabled"]):
			return
		gene["enabled"] = False
	
		gene1 = self._copy_gene(gene)
		gene1["out"] = genome["maxneuron"]
		gene1["weight"] = 1.0
		gene1["innovation"] = self._new_innovation()
		gene1["enabled"] = True
		genome["genes"].append(gene1)
	
		gene2 = self._copy_gene(gene)
		gene2["into"] = genome["maxneuron"]
		gene2["innovation"] = self._new_innovation()
		gene2["enabled"] = True
		genome["genes"].append(gene2)

	def _copy_gene(self, gene):
		gene2 = self._new_gene()
		gene2["into"] = gene["into"]
		gene2["out"] = gene["out"]
		gene2["weight"] = gene["weight"]
		gene2["enabled"] = gene["enabled"]
		gene2["innovation"] = gene["innovation"]
	
		return gene2

	def _point_mutate(self, genome):
		step = genome["mutationRates"]["step"]
	
		for gene in genome["genes"]:
			if (np.random.rand() < self.PERTURB_CHANCE):
				gene["weight"] = gene["weight"] + np.random.rand() * step * 2 - step
			else:
				gene["weight"] = np.random.rand() * 4 - 2

	def _link_mutate(self, genome, force_bias):
		neuron1 = self._random_neuron(genome["genes"], False)
		neuron2 = self._random_neuron(genome["genes"], True)
	 
		new_link = self._new_gene()
		if (neuron1 <= self.INPUTS and neuron2 <= self.INPUTS):
			# Both input nodes
			return

		if (neuron2 <= self.INPUTS):
			# Swap output and input
			temp = neuron1
			neuron1 = neuron2
			neuron2 = temp

		new_link["into"] = neuron1
		new_link["out"] = neuron2
		if (force_bias):
			new_link["into"] = self.INPUTS
	
		if (self._contains_link(genome["genes"], new_link)):
			return

		new_link["innovation"] = self._new_innovation()
		new_link["weight"] = np.random.rand() * 4 - 2

		genome["genes"].append(new_link)

	def _random_neuron(self, genes, non_input):
		neurons = dict()
		if (not non_input):
			for i in range(self.INPUTS):
				neurons[i] = True

		for o in range(self.OUTPUTS):
			neurons[self.MAX_NODES+o] = True

		for i in range(len(genes)):
			if ((not non_input) or genes[i]["into"] > self.INPUTS):
				neurons[genes[i]["into"]] = True

			if ((not non_input) or genes[i]["out"] > self.INPUTS):
				neurons[genes[i]["out"]] = True

		count = 0
		for x,y in neurons.items():
			count = count + 1

		n = np.random.randint(1,count+1)
		
		for k,v in neurons.items():
			n = n - 1
			if (n == 0):
				return k
		
		return 0

	def _new_gene(self):
		gene = dict()
		gene["into"] = 0
		gene["out"] = 0
		gene["weight"] = 0.0
		gene["enabled"] = True
		gene["innovation"] = 0
	
		return gene

	def _contains_link(self, genes, link):
		for gene in genes:
			if (gene["into"] == link["into"] and gene["out"] == link["out"]):
				return True
		return False

	def _new_innovation(self):
		self.pool["innovation"] = self.pool["innovation"] + 1
		return self.pool["innovation"]

	def _add_to_species(self, child):
		found_species = False
		for species in self.pool["species"]:
			if (not found_species and self._same_species(child, species["genomes"][0])):
				species["genomes"].append(child)
				found_species = True
	
		if (not found_species):
			child_species = self._new_species()
			child_species["genomes"].append(child)
			self.pool["species"].append(child_species)

	def _new_species(self):
		species = dict()
		species["topFitness"] = 0
		species["staleness"] = 0
		species["genomes"] = []
		species["averageFitness"] = 0
	
		return species

	def _same_species(self, genome1, genome2):
		dd = self.DELTA_DISJOINT * self._disjoint(genome1["genes"], genome2["genes"])
		dw = self.DELTA_WEIGHTS * self._weights(genome1["genes"], genome2["genes"]) 
		return (dd + dw < self.DELTA_THRESHOLD)

	def _disjoint(self, genes1, genes2):
		i1 = dict()
		for gene in genes1:
			i1[gene["innovation"]] = True

		i2 = dict()
		for gene in genes2:
			i2[gene["innovation"]] = True
		
		disjoint_genes = 0
		for gene in genes1:
			if ((gene["innovation"] in i2) and (not i2[gene["innovation"]])):
				disjoint_genes = disjoint_genes + 1
		
		for gene in genes2:
			if ((gene["innovation"] in i1) and (not i1[gene["innovation"]])):
				disjoint_genes = disjoint_genes + 1
		
		n = max(len(genes1), len(genes2))
		
		return disjoint_genes / n

	def _weights(self, genes1, genes2):
		i2 = dict()
		for gene in genes2:
			i2[gene["innovation"]] = gene

		total = 0
		coincident = 0
		for gene in genes1:
			if (gene["innovation"] in i2):
				gene2 = i2[gene["innovation"]]
				total = total + abs(gene["weight"] - gene2["weight"])
				coincident = coincident + 1
	
		if (coincident == 0):
			return float('Inf')
		return total / coincident

	def prepare_next(self):
		current_species = self.pool["currentSpecies"]
		current_genome = self.pool["currentGenome"]

		for i in range(3):
			if (current_genome >= len(self.pool["species"][current_species]["genomes"])):
				current_species = current_species + 1
				current_genome = 0

			if (current_species >= len(self.pool["species"])):
				break;

			current = self.pool["species"][current_species]["genomes"][current_genome]
			current_genome = current_genome + 1
			self._generate_network(current)

	def _generate_network(self, genome):
		network = dict()
		network["neurons"] = dict()
	
		for i in range(self.INPUTS):
			network["neurons"][i] = self._new_neuron()
	
		for o in range(self.OUTPUTS):
			network["neurons"][self.MAX_NODES+o] = self._new_neuron()
	
		genome["genes"].sort(key=lambda x : x["out"])

		for gene in genome["genes"]:
			if (gene["enabled"]):
				if (not (gene["out"] in network["neurons"])):
					network["neurons"][gene["out"]] = self._new_neuron()
				neuron = network["neurons"][gene["out"]]
				neuron["incoming"].append(gene)
				if (not (gene["into"] in network["neurons"])):
					network["neurons"][gene["into"]] = self._new_neuron()
		
		genome["network"] = network

	def _new_neuron(self):
		neuron = dict()
		neuron["incoming"] = []
		neuron["value"] = 0.0

		return neuron

	def feed(self, inputs):
		species = self.pool["species"][self.pool["currentSpecies"]]
		genome = species["genomes"][self.pool["currentGenome"]]

		keys = self._evaluate_network(genome["network"], inputs)

		return keys

	def _evaluate_network(self, network, inputs):
		inputs = np.append(inputs, [1])

		if (len(inputs) != self.INPUTS):
			return [0,0,0,0,0]
		
		for i in range(self.INPUTS):
			network["neurons"][i]["value"] = inputs[i]
		
		for key, neuron in network["neurons"].items():
			total = 0
			for j in range(len(neuron["incoming"])):
				incoming = neuron["incoming"][j]
				other = network["neurons"][incoming["into"]]
				total = total + incoming["weight"] * other["value"]
			
			if (len(neuron["incoming"]) > 0):
				neuron["value"] = self._sigmoid(total)
		
		outputs = []
		for o in range(self.OUTPUTS):
			if (network["neurons"][self.MAX_NODES+o]["value"] > 0):
				outputs.append(1)
			else:
				outputs.append(0)
		
		return outputs

	def _sigmoid(self, x):
		return 2/(1+math.exp(-4.9*x))-1

	def fitness(self, mario_positions):
		species = self.pool["species"][self.pool["currentSpecies"]]
		genome = species["genomes"][self.pool["currentGenome"]]
		
		genome["fitness"] = 4 * mario_positions[1] - mario_positions[0]
		print(str(genome["fitness"]))

		if (genome["fitness"] > self.pool["maxFitness"]):
			self.pool["maxFitness"] = genome["fitness"]

		self.pool["currentGenome"] = self.pool["currentGenome"] + 1
		if (self.pool["currentGenome"] >= len(self.pool["species"][self.pool["currentSpecies"]]["genomes"])):
			self.pool["currentGenome"] = 0
			self.pool["currentSpecies"] = self.pool["currentSpecies"] + 1

		if (self.pool["currentSpecies"] >= len(self.pool["species"])):
			self.pool["currentSpecies"] = 0
			self.generation_evaluated = True
		else :
			print("Gen " + str(self.pool["generation"]) + " - Specie " + str(self.pool["currentSpecies"]) + " - Ind " + str(self.pool["currentGenome"]) + " - Score : ", end='', flush=True)

	def generation_finished(self):
		return self.generation_evaluated

	def next_generation(self):
		print("End generation " + str(self.pool["generation"]) + " - Max fitness : " + str(self.pool["maxFitness"]))
		self._save_current_pool()
		self._cull_species(False) # Cull the bottom half of each species
		self._rank_globally()
		self._remove_stale_species()
		self._rank_globally()
		for species in self.pool["species"]:
			self._calculate_average_fitness(species)

		self._remove_weak_species()
		tot = self._total_average_fitness()
		children = []
		for species in self.pool["species"]:
			breed = math.floor(species["averageFitness"] / tot * self.POPULATION) - 1
			for i in range(breed):
				children.append(self._breed_child(species))

		self._cull_species(True) # Cull all but the top member of each species
		while (len(children) + len(self.pool["species"]) < self.POPULATION):
			species = self.pool["species"][np.random.randint(0,len(self.pool["species"]))]
			children.append(self._breed_child(species))

		for child in children:
			self._add_to_species(child)
	
		self.pool["generation"] = self.pool["generation"] + 1
		self.generation_evaluated = False
		print("Gen " + str(self.pool["generation"]) + " - Specie " + str(self.pool["currentSpecies"]) + " - Ind " + str(self.pool["currentGenome"]) + " - Score : ", end='', flush=True)

	def _cull_species(self, cut_to_one):
		for species in self.pool["species"]:
			species["genomes"].sort(key=lambda x : x["fitness"], reverse=True)
		
			remaining = math.ceil(len(species["genomes"])/2)
			if (cut_to_one):
				remaining = 1

			while (len(species["genomes"]) > remaining):
				species["genomes"].pop()

	def _rank_globally(self):
		general = []
		for species in self.pool["species"]:
			for genome in species["genomes"]:
				general.append(genome)

		general.sort(key=lambda x : x["fitness"])
	
		for g in range(len(general)):
			general[g]["globalRank"] = g+1

	def _remove_stale_species(self):
		survived = []

		for species in self.pool["species"]:
			species["genomes"].sort(key=lambda x : x["fitness"], reverse=True)
			
			if (species["genomes"][0]["fitness"] > species["topFitness"]):
				species["topFitness"] = species["genomes"][0]["fitness"]
				species["staleness"] = 0
			else :
				species["staleness"] = species["staleness"] + 1

			if (species["staleness"] < self.STALE_SPECIES or species["topFitness"] >= self.pool["maxFitness"]):
				survived.append(species)

		self.pool["species"] = survived

	def _calculate_average_fitness(self, species):
		total = 0
	
		for genome in species["genomes"]:
			total = total + genome["globalRank"]
		
		species["averageFitness"] = total / len(species["genomes"])

	def _remove_weak_species(self):
		survived = []

		total = self._total_average_fitness()
		for species in self.pool["species"]:
			breed = math.floor(species["averageFitness"] / total * self.POPULATION)
			if (breed >= 1):
				survived.append(species)

		self.pool["species"] = survived

	def _total_average_fitness(self):
		total = 0
		for species in self.pool["species"]:
			total = total + species["averageFitness"]

		return total

	def _breed_child(self, species):
		child = dict()
		if (np.random.rand() < self.CROSSOVER_CHANCE):
			g1 = species["genomes"][np.random.randint(0,len(species["genomes"]))]
			g2 = species["genomes"][np.random.randint(0,len(species["genomes"]))]
			child = self._crossover(g1, g2)
		else :
			g = species["genomes"][np.random.randint(0,len(species["genomes"]))]
			child = self._copy_genome(g)
		
		self._mutate(child)
		
		return child

	def _crossover(self, g1, g2):
		# Make sure g1 is the higher fitness genome
		if (g2["fitness"] > g1["fitness"]):
			tempg = g1
			g1 = g2
			g2 = tempg

		child = self._new_genome()
		
		innovations2 = dict()
		for gene in g2["genes"]:
			innovations2[gene["innovation"]] = gene
		
		for gene1 in g1["genes"]:
			if (gene1["innovation"] in innovations2):
				gene2 = innovations2[gene1["innovation"]]
				if (np.random.randint(1,3) == 1 and gene2["enabled"]):
					child["genes"].append(self._copy_gene(gene2))
				else :
					child["genes"].append(self._copy_gene(gene1))
		
		child["maxneuron"] = max(g1["maxneuron"],g2["maxneuron"])
		
		for mutation,rate in g1["mutationRates"].items():
			child["mutationRates"][mutation] = rate
		
		return child

	def _copy_genome(self, genome):
		genome2 = self._new_genome()

		for genes in genome["genes"]:
			genome2["genes"].append(self._copy_gene(genes))

		genome2["maxneuron"] = genome["maxneuron"]
		genome2["mutationRates"]["connections"] = genome["mutationRates"]["connections"]
		genome2["mutationRates"]["link"] = genome["mutationRates"]["link"]
		genome2["mutationRates"]["bias"] = genome["mutationRates"]["bias"]
		genome2["mutationRates"]["node"] = genome["mutationRates"]["node"]
		genome2["mutationRates"]["enable"] = genome["mutationRates"]["enable"]
		genome2["mutationRates"]["disable"] = genome["mutationRates"]["disable"]
	
		return genome2

	def _save_current_pool(self):
		name = 'plugins/SerpentDonkeyKongGameAgentPlugin/files/helpers/neat_gen_' + str(self.pool["generation"]) + '.json'
		with open(name, 'w') as fp:
			json.dump(self.pool, fp)