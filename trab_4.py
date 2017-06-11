# This Python file uses the following encoding: utf-8

# Trabalho 4 CSM
# João Filipe Vaz - 40266
# João Ventura - 38950


import cv2
import numpy as np
from time import time
from os import path


# 1

"""
Considerar que cada frame é uma intra-frame (I). Pode usar o codiﬁcador do trabalho anterior ou usar o codiﬁ-
cador já disponível
"""


def comprime():

    for i in xrange(1, 12, 1):

        cv2.imwrite("output/bola_{}.jpeg".format(i), cv2.imread("samples/bola_{}.tiff".format(i)),
                    (cv2.IMWRITE_JPEG_QUALITY, 50))
        cv2.imwrite("output/car{}.jpeg".format(i), cv2.imread("samples/car{}.bmp".format(i)),
                    (cv2.IMWRITE_JPEG_QUALITY, 50))


def cod_intraframe():

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


"""
Main
"""

def main():
    print "========================================================================================================"
    print "================================Analise Ficheiro lena================================" \
          "======="



    t1 = time()
    print "O tempo necessário para efectuar a DCT e a Quantificação foi de {} segundos".format(round(t1 - t0, 3))



    t2 = time()
    print "O tempo necessário para efectuar a codificação DC foi de {} segundos".format(round(t2 - t1, 3))



    t3 = time()
    print "O tempo necessário para efectuar a codificação AC foi de {} segundos".format(round(t3 - t2, 3))



    t4 = time()
    print "O tempo necessário para o bloco de entropy coding (huffman) foi de {} segundos".format(round(t4 - t3, 3))



    t5 = time()
    print "O tempo necessário para a leitura do ficheiro e reconstrução do ac e dc foi de {} " \
          "segundos".format(round(t5 - t4, 3))


    t6 = time()
    print "O tempo necessário para a descodificacao ac foi de {} segundos".format(round(t6 - t5, 3))



    t7 = time()
    print "O tempo necessário para a descodificacao dc foi de {} segundos".format(round(t7 - t6, 3))


    t8 = time()
    print "O tempo necessário para a dct inversa dc foi de {} segundos".format(round(t8 - t7, 3))


    print "factor q = " + str(q)
    print "alfa = " + str(alfa)
    print "SNR = " + str(calculoSNR(x, x_rec))
    size_ini = path.getsize("samples/lena.tiff")
    size_end = path.getsize("Lena descodificada Q = {}.jpeg".format(q))
    print "A dimensão do ficheiro original é de {} Kb".format(round(size_ini / 1024., 2))
    print "A dimensão do ficheiro codificado é de {} Kb".format(round(size_end / 1024., 2))
    print "A taxa de compressão conseguida foi de {}".format(1. * size_ini / size_end)
    print "O saldo da compressão foi de {} Kb".format(round((size_ini - size_end) / 1024., 2))

    print "========================================================================================================"
    print "========================================================================================================"
    print "========================================================================================================"
    print
    print

main()
