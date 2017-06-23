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
import sys


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
        print "A Energia media por bit é {}".format()

        # Calcula o histogram
        h, bins, patches = plt.hist(x_bola_desc.ravel(), np.max(x_bola_desc), [0, np.max(x_bola_desc)])

        entropia(x_bola_desc.ravel(), gera_huffman(h))
        print "========================================================================================================"


def converte_para_video_intra():

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

    # leitura imagem bola
    i_frame = cv2.imread("samples/bola_1.tiff")

    cv2.imwrite("output/bola_iframe_1.jpeg", i_frame, (cv2.IMWRITE_JPEG_QUALITY, 50))

    for i in xrange(1, 12, 1):

        x_bola = cv2.imread("samples/bola_{}.tiff".format(i))

        if i > 1:
            t0 = time()

            p_frame = cv2.imread("samples/bola_{}.tiff".format(i)) - i_frame + 128

            # leitura imagem car
            # x_car = cv2.imread("samples/car{}.bmp".format(i))

            cv2.imwrite("output/bola_pframe_{}.jpeg".format(i), p_frame, (cv2.IMWRITE_JPEG_QUALITY, 50))

            t1 = time()
            print "O tempo necessário para efectuar a compressão e descompressão da frame {} foi de {} segundos".format(i,
                                        round(t1 - t0, 4))

            # conversão e escrita com factor de qualidade 50
            # x_car_desc = cv2.imwrite("output/car{}.jpeg".format(i), x_car, (cv2.IMWRITE_JPEG_QUALITY, 50))

            print "ANALISE FRAME " + str(i)
            # calculo SNR bola
            print "SNR = " + str(calculoSNR(x_bola, p_frame))

            size_ini = path.getsize("samples/bola_{}.tiff".format(i))
            size_end = path.getsize("output/bola_pframe_{}.jpeg".format(i))
            print "A dimensão da frame original é de {} Kb".format(round(size_ini / 1024., 2))
            print "A dimensão da frame codificada é de {} Kb".format(round(size_end / 1024., 2))
            print "A taxa de compressão conseguida foi de {}".format(1. * size_ini / size_end)

            # Calcula o histogram
            h, bins, patches = plt.hist(p_frame.ravel(), np.max(p_frame), [0, np.max(p_frame)])

            entropia(p_frame.ravel(), gera_huffman(h))
        print "========================================================================================================"

    def converte_para_video_inter():

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        bola = cv2.VideoWriter('bola_inter.avi', fourcc, 20, (352, 240))
        # car = cv2.VideoWriter('car_inter.avi', fourcc, 20, (256, 256))

        iframe = cv2.imread("output/bola_iframe_1.jpeg")

        bola.write(iframe)

        for i in xrange(2, 12, 1):
            pframe = cv2.imread("output/bola_pframe_{}.jpeg".format(i)) - 128

            bola.write(iframe + pframe)
            # car.write(cv2.imread("output/car{}.jpeg".format(i)))

        bola.release()
        # car.release()
        cv2.destroyAllWindows()


# 3.1

"""
 Considerar que cada frame à excepção da primeira são inter-frames (P). Neste codiﬁcador deve implementar a
predição da frame a codiﬁcar com base na I-frame fazendo a compensação de movimento. A frame a transmitir
é a diferença entre a frame a codiﬁcar e a sua predição. Sugere-se a construção de três funções:

3.1. uma função para fazer a medição do erro absoluto médio entre dois blocos (tamanho 16 × 16);
"""


def erro_abs_medio(bloco_actual, bloco_anterior):

    return np.mean(abs(bloco_actual.astype(np.float) - bloco_anterior.astype(np.float)))


# 3.2

"""
3.2. uma função que faça uma pesquisa (pode escolher a full-search ou outra) do bloco da frame a codiﬁcar
numa janela de pesquisa (-15 a +15) da I-frame;
"""


def fullsearch(frame_anterior, bloco_p_frame, pos_bloco, dim_janela_pesquisa):

    altura = frame_anterior.shape[0]
    largura = frame_anterior.shape[1]

    eam_min = sys.maxint

    motion_vector = None

    bloco_a_codificar = np.zeros((16, 16))

    lim_hor_esq = pos_bloco[0] - dim_janela_pesquisa
    lim_ver_sup = pos_bloco[1] - dim_janela_pesquisa
    lim_hor_drt = pos_bloco[0] + 16 + dim_janela_pesquisa
    lim_ver_inf = pos_bloco[1] + 16 + dim_janela_pesquisa

    if lim_hor_esq < 0:
        lim_hor_esq = 0

    if lim_hor_drt >= largura:
        lim_hor_drt = largura

    if lim_ver_sup < 0:
        lim_ver_sup = 0

    if lim_ver_inf >= altura:
        lim_ver_inf = altura

    janela_pesquisa = frame_anterior[lim_ver_sup:lim_ver_inf, lim_hor_esq:lim_hor_drt, 0]

    if janela_pesquisa.shape < (31, 31):
        print janela_pesquisa.shape

        print "lim_ver_sup " + str(lim_ver_sup)
        print "lim_ver_inf " + str(lim_ver_inf)

        print "lim_hor_esq " + str(lim_hor_esq)
        print "lim_hor_drt " + str(lim_hor_drt)

    if janela_pesquisa.shape >= (31, 31):

        for x in xrange(janela_pesquisa.shape[1] - dim_janela_pesquisa):
            for y in xrange(janela_pesquisa.shape[0] - dim_janela_pesquisa):
                i_bloco = janela_pesquisa[y:y + 16, x:x + 16]

                eam = erro_abs_medio(bloco_p_frame, i_bloco)
                if eam < eam_min:
                    eam_min = eam
                    bloco_a_codificar = i_bloco
                    motion_vector = ((lim_hor_esq + x) - pos_bloco[0], (lim_ver_sup + y) - pos_bloco[1])

    return eam_min, motion_vector, bloco_a_codificar


# 3.3

"""
3.3. uma função que percorra os blocos da frame a codiﬁcar e construa a frame predita;
Visualizar a frame predita, e a frame diferença, bem como os vectores de movimento (use a função pylab.quiver
para o efeito).
"""


def block_motion_compensation():

    print "Analise INTER-FRAME com BLOCK MOTION COMPENSATION"
    print "========================================================================================================"

    dim_janela_pesquisa = 15

    i_frame = cv2.imread("samples/bola_1.tiff")

    for i in xrange(1, 12, 1):

        if i > 1:
            t0 = time()

            p_frame = cv2.imread("samples/bola_{}.tiff".format(i))

            altura = p_frame.shape[0]
            largura = p_frame.shape[1]

            frame_diferenca = np.zeros((altura, largura))

            frame_predita = np.zeros((altura, largura))

            n_blocos_horizontais = largura / 16
            n_blocos_verticais = altura / 16

            xy_motion = np.zeros((n_blocos_horizontais, n_blocos_verticais))

            for x in xrange(n_blocos_horizontais):
                for y in xrange(n_blocos_verticais):
                    bloco_p_frame = p_frame[(y * 16):16 + (y * 16), (x * 16):16 + (x * 16), 0]

                    eam_min, motion_vector, bloco_a_codificar = fullsearch(i_frame, bloco_p_frame,
                                                                           ((x * 16), (y * 16)), dim_janela_pesquisa)

                    # bloco diferenca ou erro entre a p_frame original e a sua predição
                    bloco_diferenca = bloco_p_frame.astype(np.float) - bloco_a_codificar.astype(np.float)

                    xy_motion[x, y] = motion_vector

                    # cria frame diferenca
                    frame_diferenca[(y * 16):16 + (y * 16), (x * 16):16 + (x * 16)] = bloco_diferenca

                    # frame predita é a bloco estimado mais o erro
                    frame_predita[(y * 16):16 + (y * 16), (x * 16):16 + (x * 16)] = bloco_a_codificar + bloco_diferenca

            # p_frame = cv2.imread("samples/bola_{}.tiff".format(i)) - i_frame + 128

            # leitura imagem car
            # x_car = cv2.imread("samples/car{}.bmp".format(i))

            cv2.imwrite("output/bola_pframe_predita_{}.jpeg".format(i), frame_predita, (cv2.IMWRITE_JPEG_QUALITY, 50))

            cv2.imwrite("output/bola_pframe_diferenca_{}.jpeg".format(i), frame_diferenca,
                        (cv2.IMWRITE_JPEG_QUALITY, 50))

            # X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
            #X = [i for ]
            #Y = np.linspace(1, altura, 50)
            #U = xy_motion[0]
            #V = xy_motion[1]

            #plt.title('Arrows scale with plot width, not view')
            #Q = plt.quiver(U, V, units='width')
            #qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
            #                   coordinates='figure')

            #plt.show()

            t1 = time()
            print "O tempo necessário para efectuar a compressão e descompressão da frame {} foi de {} segundos".format(i,
                                        round(t1 - t0, 4))

            # conversão e escrita com factor de qualidade 50
            # x_car_desc = cv2.imwrite("output/car{}.jpeg".format(i), x_car, (cv2.IMWRITE_JPEG_QUALITY, 50))

            print "ANALISE FRAME " + str(i)
            # calculo SNR bola
            print "SNR = " + str(calculoSNR(i_frame, p_frame))

            size_ini = path.getsize("samples/bola_{}.tiff".format(i))
            size_end = path.getsize("output/bola_pframe_{}.jpeg".format(i))
            print "A dimensão da frame original é de {} Kb".format(round(size_ini / 1024., 2))
            print "A dimensão da frame codificada é de {} Kb".format(round(size_end / 1024., 2))
            print "A taxa de compressão conseguida foi de {}".format(1. * size_ini / size_end)

            # Calcula o histogram
            h, bins, patches = plt.hist(p_frame.ravel(), np.max(p_frame), [0, np.max(p_frame)])

            entropia(p_frame.ravel(), gera_huffman(h))
        print "========================================================================================================"



"""
Funções Auxiliares
"""


# função auxiliar para calcular o SNR entre a imagem original e a comprimida
def calculoSNR(imgOrig, imgComp):
    PSinal = np.sum(imgComp**2.0)
    PRuido = np.sum((imgOrig-imgComp)**2.0)
    args = PSinal / PRuido
    return np.round(10.0*np.log10(args), 3)

def energia_media_bit(img_a_transmitir):

    emb = 0
    altura = img_a_transmitir.shape[0]
    largura = img_a_transmitir.shape[1]
    for x in xrange(largura):
        for y in xrange(altura):
            emb += np.abs(img_a_transmitir[y][x]**2.)

    return emb / (altura * largura)


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
    elif coding == "block":
        block_motion_compensation()

    print "========================================================================================================"
    print "========================================================================================================"
    print "========================================================================================================"
    print
    print

main("block")
