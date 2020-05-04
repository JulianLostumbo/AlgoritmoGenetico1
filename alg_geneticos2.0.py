import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from operator import itemgetter
def inicializar(tamaño_poblacion, longitud_cromosoma):
    poblacion_inicial = []
    for i in range(tamaño_poblacion):
        num = []
        for i in range(longitud_cromosoma):
            num.append(str(np.random.randint(0,2)))
        b = "".join(num)
        poblacion_inicial.append(b)
    return poblacion_inicial
def seleccionar_ruleta_casera(poblacion, elitismo=False):
    lista_poblacion = []
    padres = []
    for id, individuo in enumerate(poblacion):
        lista_poblacion.append([id, individuo, fitness(individuo)])
        
    lista_poblacion = sorted(lista_poblacion, key=itemgetter(2), reverse=True)
    
    if elitismo:
        padres.append(lista_poblacion[0][1])
        padres.append(lista_poblacion[1][1])
    else:
        # RULETA 
        limite_superior = np.sum([x[2] for x in lista_poblacion]) # suma de las fitness

        for k in range(2):
            num_aleatorio = np.random.rand() * limite_superior    
            fitness_acumulada = lista_poblacion[0][2]
            i = 0
            
            while(fitness_acumulada < limite_superior):
                if num_aleatorio <= fitness_acumulada:
                    id_seleccionado = lista_poblacion[i][0]
                    break
                else: 
                    i += 1
                    fitness_acumulada += lista_poblacion[i][2]
                    

            if fitness_acumulada == limite_superior:
                id_seleccionado = lista_poblacion[-1][0]

            for sublista in lista_poblacion:
                if sublista[0] == id_seleccionado:
                    cromosoma_elegido = sublista[1]
                    break
            padres.append(cromosoma_elegido)
            
    return padres[0], padres[1]
def seleccionar(poblacion, elitismo=False):
    if elitismo:
        individuo_puntaje = []
        
        for individuo in poblacion:
            individuo_puntaje.append((individuo, fitness(individuo)))
        individuo_puntaje = sorted(individuo_puntaje, key=itemgetter(1), reverse=True) ## Ordena según fitness
        
        padre_1 = individuo_puntaje[0][0]
        padre_2 = individuo_puntaje[1][0]
        
    else:
        puntajes = [fitness(individuo) for individuo in poblacion]
        probabilidades = np.array(puntajes) / np.sum(puntajes)
        padre_1 = np.random.choice(poblacion, p = probabilidades)
        padre_2 = np.random.choice(poblacion, p = probabilidades)
        
    return padre_1, padre_2
def fitness(individuo):
    individuo_decimal = int(individuo, 2) 
    return individuo_decimal / (2**30 - 1)
def mutar(cromosoma):
    posicion = np.random.randint(0, len(cromosoma))
    cromosoma = list(cromosoma)
    if cromosoma[posicion] == '0':
        cromosoma[posicion] = '1'
    else:
        cromosoma[posicion] = '0'
    cromosoma = ''.join(cromosoma)
    return cromosoma
# TEST: hijo_1, hijo_2 = crossover('11111111', '00000000')
def crossover(padre_1, padre_2):
    punto_corte = np.random.randint(1, len(padre_1))
    hijo_1 = padre_1[:punto_corte] + padre_2[punto_corte:]
    hijo_2 = padre_2[:punto_corte] + padre_1[punto_corte:]
    return hijo_1, hijo_2
def generar_estadisticos(poblaciones):
    # -----------------------------------------------------------------------------------------
    # GENERACIÓN DE TABLA
    df = pd.DataFrame(columns=['Poblacion', 'Mínimo', 'Máximo', 'Media', 'Mejor cromosoma'])
    minimos = []
    maximos = []
    medias = []
    mejores_cromosomas = []
    ids = [x for x in range(len(poblaciones))]
    
    for poblacion in poblaciones:
        puntajes = []
        
        for individuo in poblacion:
            puntajes.append(fitness(individuo))

        minimos.append(np.min(puntajes))
        maximos.append(np.max(puntajes))
        medias.append(np.mean(puntajes))
        
        max_puntaje = 0
        id_puntaje = 0
        for i, puntaje in enumerate(puntajes):
            if puntaje > max_puntaje:
                max_puntaje = puntaje
                id_puntaje = i
        mejores_cromosomas.append(poblacion[id_puntaje])
        
    df['Poblacion'] = ids
    df['Mínimo'] = minimos
    df['Máximo'] = maximos
    df['Media'] = medias
    df['Mejor cromosoma'] = mejores_cromosomas
    df = df.set_index('Poblacion')
    df = df.round(decimals = 3)
    # ----------------------------------------------------------------------------------------------
    # GENERACIÓN DE GRÁFICOS
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize = (12, 7), sharex = True)
    
    ax0.plot(ids, medias, label='Media', linewidth=1.7, color="firebrick")
    ax1.plot(ids, minimos, label='Mínimo', linewidth=1.7, color="blue")
    ax2.plot(ids, maximos, label='Máximo', linewidth=1.7, color="green")

    ax0.set_title('Evolución de las medias ') 
    ax0.set_ylabel('Media')
    ax0.set_xlim(0,len(ids) - 1)
    ax0.grid(True)
    ax0.legend(loc='lower right')
    
    ax1.set_title('Evolución de los mínimos') 
    ax1.set_ylabel('Mínimo')
    ax1.set_xlim(0,len(ids) - 1)
    ax1.grid(True)
    ax1.legend(loc='lower right')
    
    ax2.set_title('Evolución de los máximos') 
    ax2.set_ylabel('Máximo')
    ax2.set_xlim(0,len(ids) - 1)
    ax2.set_xlabel('Número de generación')
    ax2.grid(True)
    ax2.legend(loc='lower right')
    ax2.set_xticks(ids)
    
    
    plt.rcParams.update({'font.size': 12})
    plt.tight_layout()
#   plt.savefig(f'{image_directory}/learning_curves/Trials curves - Training & Validation plots - {base_study} - {training_dataset} - {validation_dataset}.pdf')
    plt.show()
    
    return df
def main(p_crossover=0.75, p_mutacion=0.05, ciclos=20, tamaño_poblacion=10, longitud_cromosoma=30, elitismo=True):
    poblacion_actual = inicializar(tamaño_poblacion, longitud_cromosoma)
    poblaciones = []
    poblaciones.append(poblacion_actual)
    cantidad_selecciones = tamaño_poblacion // 2
    for i in range(ciclos):
        poblacion_nueva = []
        
        if elitismo:
            individuo_elite_1, individuo_elite_2 = seleccionar(poblacion_actual, elitismo)
            poblacion_nueva.append(individuo_elite_1)
            poblacion_nueva.append(individuo_elite_2)
            
        for j in range(tamaño_poblacion // 2 + 1):
         
            if len(poblacion_nueva) == tamaño_poblacion:
                continue
                
            # SELECCIONAR
            padre_1, padre_2 = seleccionar(poblacion_actual)

            # CRUZAR
            if np.random.rand() <= p_crossover:
                hijo_1, hijo_2 = crossover(padre_1, padre_2)
            else: 
                hijo_1 = padre_1
                hijo_2 = padre_2

            # MUTAR
            if np.random.rand() <= p_mutacion:
                hijo_1 = mutar(hijo_1)
            if np.random.rand() <= p_mutacion:
                hijo_2 = mutar(hijo_2)

            # PASAR A PROXIMA GENERACION
            poblacion_nueva.append(hijo_1)
            if len(poblacion_nueva) == tamaño_poblacion:
                continue
            else:
                poblacion_nueva.append(hijo_2)

        poblacion_actual = poblacion_nueva
        
        poblaciones.append(poblacion_nueva)

    poblaciones_df = pd.DataFrame(poblaciones)
    resultados = generar_estadisticos(poblaciones)

    print(resultados)
p_crossover = 0.75
p_mutacion = 0.05
ciclos = 20
tamaño_poblacion = 10
longitud_cromosoma = 30

main(p_crossover, p_mutacion, ciclos, tamaño_poblacion, longitud_cromosoma, elitismo=True)