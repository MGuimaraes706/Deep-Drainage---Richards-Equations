import numpy as np

# ==============================================================================
# 1. MAPEAMENTO DE BORDAS (Tags das Linhas do Gmsh)
# ==============================================================================
# Estes números vêm do arquivo .xdmf gerado pelo converter_malha.py
TAG_BORDA_ESQUERDA = 101
TAG_BORDA_DIREITA = 102
TAG_BORDA_BASE = 103
TAG_BORDA_TOPO = 104
TAG_PAREDE_RIO = 105

# ==============================================================================
# 2. PARÂMETROS DO CENÁRIO HIDRÁULICO (RIO)
# ==============================================================================
VAZAO_RIO = 2.0         # m3/s
MANNING_N = 0.013       # Concreto liso
DECLIVIDADE_S0 = 0.005  # 0.5%
LARGURA_CANAL = 2.0     # m (w_int do Gmsh)

# IMPORTANTE: Cota do fundo da aduela na malha (conforme o arquivo .geo)
# Se no .geo y_aduela = 1.0, aqui deve ser 1.0.
Y_FUNDO_ADUELA = 1.0    

# ==============================================================================
# 3. CÁLCULO DO NÍVEL DO RIO
# ==============================================================================
def calcular_nivel_rio():
    """
    Calcula a altura 'h' (lâmina d'água relativa ao fundo) usando Manning.
    """
    # Valor aproximado para Q=2.0, n=0.013, S=0.005, b=2.0
    return 0.78 

# Altura da Água relativa ao fundo do concreto
LAMINA_AGUA = calcular_nivel_rio()

# Nível Absoluto na Malha (Y Global)
# CORREÇÃO: Somamos a cota do fundo. Se não fizer isso, o rio fica "enterrado".
NIVEL_AGUA_ABSOLUTO = Y_FUNDO_ADUELA + LAMINA_AGUA

# ==============================================================================
# 4. FUNÇÕES DE CONDIÇÃO DE CONTORNO
# ==============================================================================

def aplicar_bc_externa(x, y, tag_borda):
    """Define o valor de Dirichlet (Carga h) para bordas externas."""
    
    # ESQUERDA: Lençol Freático Alto (Entrada)
    if tag_borda == TAG_BORDA_ESQUERDA:
        nivel_freatico = 4.0
        return nivel_freatico - y 

    # DIREITA: Lençol Freático Mais Baixo (Saída)
    elif tag_borda == TAG_BORDA_DIREITA:
        nivel_freatico = 3.5
        return nivel_freatico - y

    # BASE: Impermeável
    elif tag_borda == TAG_BORDA_BASE:
        return None # Indica Neumann (Fluxo zero)

    # TOPO: Seco/Atmosférico
    elif tag_borda == TAG_BORDA_TOPO:
        return -1.0 # Leve sucção
    
    return None

def condicao_parede_rio(y_coord):
    """
    Define a pressão na parede interna do duto.
    """
    # CORREÇÃO: Compara a coordenada do nó com o Nível Absoluto (1.78m)
    
    # 1. ZONA SUBMERSA
    if y_coord < NIVEL_AGUA_ABSOLUTO:
        # P = (Cota da Superfície) - (Cota do Ponto)
        pressao = NIVEL_AGUA_ABSOLUTO - y_coord
        tipo = "Dirichlet"
        return tipo, pressao
    
    # 2. ZONA SECA (Acima da água)
    else:
        # Pressão 0 (Atmosférica), sujeita à Seepage Face
        pressao = 0.0
        tipo = "SeepageFace"
        return tipo, pressao