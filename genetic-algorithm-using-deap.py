from deap import creator, tools, algorithms, base
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# objetivo: minimização
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

# indivíduos em estrutura de lista
creator.create('EstrIndividuo',list, fitness=creator.FitnessMin)

# numero de genes
n = 4

# registros dos elementos do AG
toolbox = base.Toolbox()
toolbox.register('Genes',np.random.choice,n)
toolbox.register('Individuos', tools.initIterate, creator.EstrIndividuo, toolbox.Genes)
toolbox.register('Populacao', tools.initRepeat, list, toolbox.Individuos)
pop = toolbox.Populacao(n=10)

# operadores de cruzamento e mutacao
toolbox.register('mate',tools.cxPartialyMatched) # crossover
toolbox.register('mutation',tools.mutShuffleIndexes, indpb=0.1) # mutacao
toolbox.register('select',tools.selTournament,tournsize=2)

dist = [
    [0,7,9,2],
    [4,0,3,7],
    [6,7,0,8],
    [2,3,8,0]
]

# função a ser minimizada
def aptidao(individuo):
    f=0
    for i in range(n-1):
        local1 = individuo[i]
        local2 = individuo[i+1]
        distancia = dist[local1][local2]
        return f + distancia

toolbox.register('evaluate',aptidao)

def estatisticaSalvar(individual):
    return individual.fitness.values

estatistica = tools.Statistics(estatisticaSalvar)

# salvar nas estatisticas o mínimo, o máximo e a média
estatistica.register('mean', np.mean)
estatistica.register('min', np.min)
estatistica.register('max', np.max)

# melhor solução
melhor = tools.HallOfFame(1)

# executando o AG com os parâmetros anteriormente definidos
result, log = algorithms.eaSimple(
    pop,                # população
    toolbox,            # 'caixa de ferramentas' com os elementos a serem utilizados no AG
    cxpb=0.8,         # probabilidade de cruzamento
    mutpb=0.1,        # probabilidade de mutação
    stats=estatistica,# estatisticas
    ngen=30,          # numero de gerações
    halloffame=melhor,# solução
    verbose=True      # imprimir resultados ao longo das gerações
)