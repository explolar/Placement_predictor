import numpy as np
import random

class CareerOptimizer:
    def __init__(self, model, feature_names, actionable_features, constraints):
        self.model = model
        self.feature_names = feature_names
        self.actionable = actionable_features
        self.constraints = constraints
        self.population_size = 50
        self.generations = 30

    def _get_fitness(self, individual, original):
        prob = self.model.predict_proba(individual.reshape(1, -1))[0][1]
        
        # Penalize increasing backlogs hard
        # If new backlogs > original backlogs, fitness = -100
        if 'backlogs' in self.actionable:
            b_idx = self.feature_names.index('backlogs')
            if individual[b_idx] > original[b_idx]:
                 return -1000, prob

        if prob < 0.5:
            return prob * 1000, prob 
            
        effort = 0
        for feat in self.actionable:
            idx = self.feature_names.index(feat)
            range_val = self.constraints[feat][1] - self.constraints[feat][0]
            change = abs(individual[idx] - original[idx])
            effort += (change / range_val) * 10 
            
        return (prob * 1000) - effort, prob

    def optimize(self, student_vector):
        population = []
        
        # Initialize
        for _ in range(self.population_size):
            ind = student_vector.copy()
            # Randomize
            feat = random.choice(self.actionable)
            idx = self.feature_names.index(feat)
            low, high = self.constraints[feat]
            
            # LOGIC FIX 1: Don't initialize backlogs higher than current
            if feat == 'backlogs':
                if ind[idx] > 0:
                    ind[idx] = random.randint(0, int(ind[idx]))
            else:
                ind[idx] = random.uniform(low, high)
            population.append(ind)

        best_solution = None
        best_prob = 0.0

        for gen in range(self.generations):
            scored_pop = []
            for ind in population:
                fit, prob = self._get_fitness(ind, student_vector)
                scored_pop.append((fit, ind, prob))
            
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            
            if scored_pop[0][2] > best_prob:
                best_prob = scored_pop[0][2]
                best_solution = scored_pop[0][1]
            
            survivors = [x[1] for x in scored_pop[:15]]
            new_pop = survivors[:]
            while len(new_pop) < self.population_size:
                parent = random.choice(survivors)
                child = parent.copy()
                
                # Mutation
                if random.random() < 0.4: 
                     feat = random.choice(self.actionable)
                     idx = self.feature_names.index(feat)
                     low, high = self.constraints[feat]
                     
                     # LOGIC FIX 2: Strict constraint on backlogs
                     if feat == 'backlogs':
                         current_val = int(child[idx])
                         if current_val > 0:
                             child[idx] = random.randint(0, current_val) # Can only reduce
                     else:
                         child[idx] = np.clip(child[idx] + random.randint(-5, 5), low, high)
                new_pop.append(child)
            population = new_pop
            
        return best_solution, best_prob