# Classe principal do código
# 
# exec -> python runTests.py caminha_ate_a_pasta_do_projeto
# 
# <b>runTests</b>: Esse script controla a execução de todas as partes do algoritmo.
# É nele que são lidas as imagens e estas são passadas para as demais etapadas do algoritmo
# Ao término no processo, ele gera arquivos .csv (tabelas) com os resultados individuais de cada imagem
# e a média geral de todas as execuções.
#
# Explicando resumidamente o código:
# 
#
# author:
# Pablo Vinicius - https://github.com/pabloVinicius
#
# contributors:
# Filipe A. Sampaio - https://github.com/filipeas

# imports dos arquivos necessários
from main import vsf # responsável pela separação dos superpixels
from features_extraction import ftet # responsável por extrair as caracteristicas de cada superpixel usadas no modelo
from select_and_classify import classify, select_random_seeds # responsável pela classificação do modelo criado
from statistics import mean, pvariance
import os, csv, time
import numpy as np
import sys

def index(original, marcada):
    # Porcentagens de dados das imagens usadas para treino.
    percentages = [
        # 0.01,
        # 0.05,
        # 0.1,
        # 0.15,
        # 0.2,
        0.25,
        # 0.3,
        # 0.35,
        # 0.4,
        # 0.5
    ]

    # Quantidade de segmentos para o superpixel
    qtdSegments = [
        1500,
        # 2000,
        # 2500,
        # 3000,
        # 4000,
        # 5000
    ]

    # Tenta encontrar o diretório results. Se ele não existir, o cria.
    results_path = 'results/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Cria a estrutura de dados que irá guardar os resultados das médias das execuções para todas as imagens e para todas as métricas.
    # Funciona como uma matriz  de (Nª de porcentagenx utilizadas)x(número de métricas, que no caso são 4).
    # No final, essa estrutura é utilizada para calcular a média das médias de cada porcentagem de dados pra treino.
    total_means = [[[],[],[],[]] for i in range(len(percentages))]

    # instanciando variavel pra guardar todas as acuracias da imagem corrente
    acuracia_corrent = [[[],[],[],[],[]] for i in range(len(percentages))]

    # instanciando variavel pra guardar todas as sensibilidade da imagem corrente
    sensibilidade_corrent = [[[],[],[],[],[]] for i in range(len(percentages))]

    # instanciando variavel pra guardar todas as especificidade da imagem corrente
    especificidade_corrent = [[[],[],[],[],[]] for i in range(len(percentages))]

    # instanciando variavel pra guardar todas as dice da imagem corrente
    dice_corrent = [[[],[],[],[],[]] for i in range(len(percentages))]

    print(f'\n\n########### Executando a imagem ############\n\n')

    # Marcando o tempo de execução.
    start = time.time()

    # Abrindo a imagem original e a imagem marcada.
    path = f'{original}' # original
    path_marked = f'{marcada}' # marcada

    print(path + '\n' + path_marked)

    # iterando sobre a quantidade de segmentos de imagem para o superpixel
    for segment in qtdSegments:
        # Executa a primeira etapa do algoritmo, responsável pela separação dos superpixels da imagem (arquivo main.py).
        vsf(path, path_marked, segment)

        # Executa a segunda etapa do algoritmo, responsável pela extração de características de cada superpixel gerado na etapa anterior.
        # Disponível no arquivo features_extraction.py
        ftet()

        # Para cada percentual:
        for index, percent in enumerate(percentages):
            
            # Cria uma estrutura de dados para salvar os resultados e posteriormente calcular as médias das métricas
            metrics_media = [[], [], [], []]

            # Executa 5 vezes para depois tirar a média (5 vezes pois são 5 métricas diferentes)
            for i in range(1):

                # Terceira etapa do algoritmo:
                # Usa as características extraídas na etapa anterior para classificar e gerar as métricas para cada imagem
                # Disponível no arquivo select_and_classify.py
                acc, sen, spe, dice = classify(percent) # classificação por Random Forest
                # acc, sen, spe, dice = select_random_seeds(image, percent) # classificação por sfc-means

                # Coloca as métricas na estrutura de dados
                metrics_media[0].append(acc)
                metrics_media[1].append(sen)
                metrics_media[2].append(spe)
                metrics_media[3].append(dice)

            # Depois adiciona na estrutura de dados os dados gerais        
            total_means[index][0].append(mean(metrics_media[0])*100) # acuracia
            total_means[index][1].append(mean(metrics_media[1])*100) # sensibilidade
            total_means[index][2].append(mean(metrics_media[2])*100) # especificidade
            total_means[index][3].append(mean(metrics_media[3])) # dice

            if(segment == 1500):
                acuracia_corrent[index][0].append(mean(metrics_media[0])*100)
                sensibilidade_corrent[index][0].append(mean(metrics_media[1])*100)
                especificidade_corrent[index][0].append(mean(metrics_media[2])*100)
                dice_corrent[index][0].append(mean(metrics_media[3])*100)
            elif(segment == 2000):
                acuracia_corrent[index][1].append(mean(metrics_media[0])*100)
                sensibilidade_corrent[index][1].append(mean(metrics_media[1])*100)
                especificidade_corrent[index][1].append(mean(metrics_media[2])*100)
                dice_corrent[index][1].append(mean(metrics_media[3])*100)
            elif(segment == 2500):
                acuracia_corrent[index][2].append(mean(metrics_media[0])*100)
                sensibilidade_corrent[index][2].append(mean(metrics_media[1])*100)
                especificidade_corrent[index][2].append(mean(metrics_media[2])*100)
                dice_corrent[index][2].append(mean(metrics_media[3])*100)
            elif(segment == 3000):
                acuracia_corrent[index][3].append(mean(metrics_media[0])*100)
                sensibilidade_corrent[index][3].append(mean(metrics_media[1])*100)
                especificidade_corrent[index][3].append(mean(metrics_media[2])*100)
                dice_corrent[index][3].append(mean(metrics_media[3])*100)
            elif(segment == 4000):
                acuracia_corrent[index][4].append(mean(metrics_media[0])*100)
                sensibilidade_corrent[index][4].append(mean(metrics_media[1])*100)
                especificidade_corrent[index][4].append(mean(metrics_media[2])*100)
                dice_corrent[index][4].append(mean(metrics_media[3])*100)

    # calcula o tempo de execução do algoritmo para a imagem atual.
    end = time.time()
    print(f'\n\n########### Fim da execucao da imagem no tempo de {end-start} segundos ############\n\n')
    print("Result = ", acuracia_corrent[0][0], sensibilidade_corrent[0][0], especificidade_corrent[0][0], dice_corrent[0][0], " Fim")
    return acuracia_corrent[0][0], sensibilidade_corrent[0][0], especificidade_corrent[0][0], dice_corrent[0][0]