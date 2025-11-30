import numpy as np

# ==============================================================================
# 1. MAPEAMENTO DE TAGS (Deve bater com o .geo/.xdmf)
# ==============================================================================
TAG_SOLO_INFERIOR = 1  # Areia
TAG_SOLO_SUPERIOR = 2  # Argila
TAG_ENVELOPE_BRITA = 3 # Brita
TAG_CONCRETO = 4       # Aduela

# ==============================================================================
# 2. PROPRIEDADES DOS MATERIAIS
# ==============================================================================
# ks: Condutividade Saturada (m/s)
# theta_s: Porosidade
# alpha, n: Parâmetros de Van Genuchten

MATERIAIS = {
    TAG_SOLO_INFERIOR: {
        "nome": "Areia Fina",
        "ks": 5.0e-5,    
        "theta_s": 0.35, 
        "alpha": 3.5,    
        "n": 2.5
    },
    TAG_SOLO_SUPERIOR: {
        "nome": "Argila Limosa",
        "ks": 1.0e-7,
        "theta_s": 0.45,
        "alpha": 0.8,
        "n": 1.4
    },
    TAG_ENVELOPE_BRITA: {
        "nome": "Brita de Drenagem",
        "ks": 1.0e-2,   # Permeabilidade muito alta
        "theta_s": 0.40,
        "alpha": 10.0,
        "n": 3.0
    },
    TAG_CONCRETO: {
        "nome": "Concreto",
        "ks": 1.0e-13,  # Impermeável
        "theta_s": 0.10,
        "alpha": 0.001,
        "n": 1.1
    }
}

def condutividade_hidraulica(h, material_tag):
    """
    Retorna o K(h) calculado via Van Genuchten.
    """
    props = MATERIAIS.get(material_tag)
    if not props:
        # Se não achar a tag, retorna um valor padrão muito baixo ou erro
        return 1e-15 
    
    ks = props['ks']
    
    if h >= 0:
        return ks
        
    alpha = props['alpha']
    n = props['n']
    m = 1.0 - (1.0 / n)
    
    # Saturação Efetiva
    Se = (1.0 + abs(alpha * h)**n)**(-m)
    
    # Mualem
    k_rel = (Se**0.5) * (1.0 - (1.0 - Se**(1.0/m))**m)**2
    return ks * k_rel