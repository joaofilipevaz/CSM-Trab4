# This Python file uses the following encoding: utf-8

# Trabalho 4 CSM
# João Filipe Vaz - 40266
# João Ventura - 38950


import cv2
import numpy as np
from time import time
import matplotlib.pyplot as plt
import Queue
from os import path


# 1

"""
Considerar que cada frame é uma intra-frame (I). Pode usar o codiﬁcador do trabalho anterior ou usar o codiﬁ-
cador já disponível
"""


def intra_frame_coding():
    print "Analise INTRA-FRAME"
    print "========================================================================================================"

    for i in xrange(1, 12, 1):
        t0 = time()

        # leitura imagem bola
        x_bola = cv2.imread("samples/bola_{}.tiff".format(i))

        # leitura imagem car
        # x_car = cv2.imread("samples/car{}.bmp".format(i))

        # conversão e escrita com factor de qualidade 50
        cv2.imwrite("output/bola_{}.jpeg".format(i), x_bola, (cv2.IMWRITE_JPEG_QUALITY, 50))

        x_bola_desc = cv2.imread("output/bola_{}.jpeg".format(i))

        t1 = time()
        print "O tempo necessário para efectuar a compressão e descompressão da frame {} foi de {} segundos".format(i,round(t1 - t0, 4))

        # conversão e escrita com factor de qualidade 50
        # x_car_desc = cv2.imwrite("output/car{}.jpeg".format(i), x_car, (cv2.IMWRITE_JPEG_QUALITY, 50))

        print "ANALISE FRAME " + str(i)
        # calculo SNR bola
        print "SNR = " + str(calculoSNR(x_bola, x_bola_desc))

        size_ini = path.getsize("samples/bola_{}.tiff".format(i))
        size_end = path.getsize("output/bola_{}.jpeg".format(i))
        print "A dimensão da frame original é de {} Kb".format(round(size_ini / 1024., 2))
        print "A dimensão da frame codificada é de {} Kb".format(round(size_end / 1024., 2))
        print "A taxa de compressão conseguida foi de {}".format(1. * size_ini / size_end)

        # Calcula o histogram
        h, bins, patches = plt.hist(x_bola_desc.ravel(), np.max(x_bola_desc), [0, np.max(x_bola_desc)])

        entropia(x_bola_desc.ravel(), gera_huffman(h))
        print "========================================================================================================"


def converte_para_video():

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    bola = cv2.VideoWriter('bola_intra.avi', fourcc, 20, (352, 240))
    car = cv2.VideoWriter('car_intra.avi', fourcc, 20, (256, 256))

    for i in xrange(1, 12, 1):

        bola.write(cv2.imread("output/bola_{}.jpeg".format(i)))
        car.write(cv2.imread("output/car{}.jpeg".format(i)))

    bola.release()
    car.release()
    cv2.destroyAllWindows()

# 2

"""
Considerar que cada frame à excepção da primeira são inter-frames (P), ou seja, todos os macroblocos são do
tipo (P). Neste codiﬁcador deve criar as P-frames, que são a diferença entre a frame a codiﬁcar e a I-frame, sem
compensação de movimento. Visualize a P-frame (ou seja a imagem a transmitir).
"""


def inter_frame_coding():
    print "Analise INTER-FRAME"
    print "========================================================================================================"

    x_bola = cv2.imread("samples/bola_1.tiff")

    # leitura imagem bola
    i_frame = cv2.imread("output/bola_1.jpeg")

    for i in xrange(2, 12, 1):
        t0 = time()

        x_bola = cv2.imread("samples/bola_{}.tiff".format(i))

        p_frame = i_frame - cv2.imread("output/bola_{}.jpeg".format(i))

        # leitura imagem car
        # x_car = cv2.imread("samples/car{}.bmp".format(i))

        cv2.imwrite("output/bola_pframe_{}.jpeg".format(i), p_frame)

        t1 = time()
        print "O tempo necessário para efectuar a compressão e descompressão da frame {} foi de {} segundos".format(i,
                                    round(t1 - t0, 4))

        # conversão e escrita com factor de qualidade 50
        # x_car_desc = cv2.imwrite("output/car{}.jpeg".format(i), x_car, (cv2.IMWRITE_JPEG_QUALITY, 50))

        print "ANALISE FRAME " + str(i)
        # calculo SNR bola
        print "SNR = " + str(calculoSNR(x_bola, p_frame))

        size_ini = path.getsize("samples/bola_{}.tiff".format(i))
        size_end = path.getsize("output/bola_{}.jpeg".format(i))
        print "A dimensão da frame original é de {} Kb".format(round(size_ini / 1024., 2))
        print "A dimensão da frame codificada é de {} Kb".format(round(size_end / 1024., 2))
        print "A taxa de compressão conseguida foi de {}".format(1. * size_ini / size_end)

        # Calcula o histogram
        h, bins, patches = plt.hist(p_frame.ravel(), np.max(p_frame), [0, np.max(p_frame)])

        entropia(p_frame.ravel(), gera_huffman(h))
        print "========================================================================================================"


# 3.1

"""
 Considerar que cada frame à excepção da primeira são inter-frames (P). Neste codiﬁcador deve implementar a
predição da frame a codiﬁcar com base na I-frame fazendo a compensação de movimento. A frame a transmitir
é a diferença entre a frame a codiﬁcar e a sua predição. Sugere-se a construção de três funções:

3.1. uma função para fazer a medição do erro absoluto médio entre dois blocos (tamanho 16 × 16);
"""


# 3.2

"""
3.2. uma função que faça uma pesquisa (pode escolher a full-search ou outra) do bloco da frame a codiﬁcar
numa janela de pesquisa (-15 a +15) da I-frame;
"""


# 3.3

"""
3.3. uma função que percorra os blocos da frame a codiﬁcar e construa a frame predita;
Visualizar a frame predita, e a frame diferença, bem como os vectores de movimento (use a função pylab.quiver
para o efeito).
"""



"""
Funções Auxiliares
"""


# função auxiliar para calcular o SNR entre a imagem original e a comprimida
def calculoSNR(imgOrig, imgComp):
    PSinal = np.sum(imgComp**2.0)
    PRuido = np.sum((imgOrig-imgComp)**2.0)
    args = PSinal / PRuido
    return np.round(10.0*np.log10(args), 3)


def gera_huffman(freq):
    # simbolos do algoritmo - neste caso uma gama de valores numa lista de listas
    simbolos = [[i] for i in xrange(len(freq))]

    # String vazia para guardar os bits resultantes da codificacao
    bits = ["" for y in xrange(len(freq))]

    # Cria um array de arrays (numero de ocorrencias,valor, codificação) organizado por ordem crescente
    tabela = sorted(list(t) for t in zip(freq, simbolos, bits))

    # remove ocorrencias nulas do array
    while tabela[0][0] == 0:
        tabela.remove(tabela[0])

    # implementa uma fila para simular o funcionamento do algoritmo
    p = Queue.PriorityQueue()

    # popula a fila com os valores do array
    for value in tabela:
        p.put(value)

    # enquanto o fila tiver mais que um nó
    while p.qsize() > 1:

        # extrai os dois nós esquerda e direita com o valor mais pequeno
        l, r = p.get(), p.get()

        # itera no array de valores e nos simbolos dos nós
        for i in xrange(len(tabela)):
            for z in xrange(len(l[1])):

                # se o simbolo no array for igual ao simbolo do nó extraido
                if tabela[i][1][0] == l[1][z]:

                    # para o no da esquerda guarda o bit 0 na codificação
                    tabela[i][2] += '0'
            for t in xrange(len(r[1])):
                if tabela[i][1][0] == r[1][t]:
                    tabela[i][2] += '1'
        # adiciona a soma dos nós a queue
        p.put([l[0] + r[0], l[1] + r[1]])

    # inverte a string com a codificação para representar o percurso da raiz até as folhas
    for i in xrange(len(tabela)):
        tabela[i][2] = tabela[i][2][::-1]

    print "=========================================Tabela Codigo=================================================="
    # imprime a array resultante em formato tabela
    print "Peso\tSimbolo\tCodificação"
    for t in range(len(tabela)):
        print "%-6s\t%-4s\t%-4s" % (tabela[t][0], tabela[t][1][0], tabela[t][2])
    print "========================================================================================================"

    return tabela


def entropia(x, tabela_codigo):

    # entropia
    hx = 0

    # numero médio de bits por simbolo
    l = 0

    for i in xrange(len(tabela_codigo)):
        prob = tabela_codigo[i][0] / len(x)
        hx += prob * np.log2(1 / prob)
        l += prob * len(tabela_codigo[i][2])

    # Eficiencia do codigo
    efic = hx / l

    print "A Entropia é de {} bits/símbolo".format(hx)
    print "O numero médio de bits por símbolo é de {}".format(l)
    print "A eficiência do código é {}".format(efic)


"""
Main
"""


def main(coding):
    print "========================================================================================================"

    if coding == "intra":
        intra_frame_coding()
    elif coding == "inter":
        inter_frame_coding()

    print "========================================================================================================"
    print "========================================================================================================"
    print "========================================================================================================"
    print
    print

main("inter")
