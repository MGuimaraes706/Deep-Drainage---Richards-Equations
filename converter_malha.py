import meshio
import os
import numpy as np

def converter_gmsh_para_xdmf(nome_arquivo_msh):
    """
    Converte um arquivo .msh (Gmsh) para dois arquivos .xdmf (Python/Paraview/FEniCS).
    Versão Robusta: Itera sobre blocos para garantir sincronia entre geometria e dados físicos.
    """
    
    # Verifica se o arquivo existe
    if not os.path.exists(nome_arquivo_msh):
        print(f"ERRO: O arquivo '{nome_arquivo_msh}' não foi encontrado.")
        print("Dica: Lembre-se de abrir o .geo no Gmsh e fazer 'File > Export > .msh'")
        return

    print(f"--> Lendo malha original: {nome_arquivo_msh}...")
    
    # Tenta ler o arquivo .msh
    try:
        msh = meshio.read(nome_arquivo_msh)
    except Exception as e:
        print(f"ERRO ao ler o arquivo: {e}")
        return

    # ===================================================
    # 1. EXTRAIR O DOMÍNIO (TRIÂNGULOS)
    # ===================================================
    print("--> Processando Triângulos (Domínio 2D)...")
    
    tri_cells = []
    tri_data = []

    # Itera sobre os blocos para garantir que pegamos o dado do bloco certo
    for i, cell_block in enumerate(msh.cells):
        if cell_block.type == "triangle":
            # Adiciona a geometria (quem conecta com quem)
            tri_cells.append(("triangle", cell_block.data))
            
            # Adiciona os dados físicos (Tags)
            if "gmsh:physical" in msh.cell_data:
                # Pega o array de dados correspondente exatamente a este bloco
                data_array = msh.cell_data["gmsh:physical"][i]
                tri_data.append(data_array)
            else:
                print("    AVISO: Bloco de triângulos sem tag física. Preenchendo com 0.")
                tri_data.append(np.zeros(len(cell_block.data)))

    if tri_cells:
        # Cria a nova malha limpa só de triângulos
        triangle_mesh = meshio.Mesh(
            points=msh.points,
            cells=tri_cells,
            cell_data={"material_tag": tri_data} # O meshio lida com a lista de arrays automaticamente
        )
        
        meshio.write("malha_dominio.xdmf", triangle_mesh)
        print("    Sucesso: 'malha_dominio.xdmf' criado.")
    else:
        print("    ERRO: Nenhum triângulo encontrado! Verifique se gerou malha 2D no Gmsh.")

    # ===================================================
    # 2. EXTRAIR AS BORDAS (LINHAS)
    # ===================================================
    print("--> Processando Linhas (Condições de Contorno)...")
    
    line_cells = []
    line_data = []

    for i, cell_block in enumerate(msh.cells):
        if cell_block.type == "line":
            line_cells.append(("line", cell_block.data))
            
            if "gmsh:physical" in msh.cell_data:
                line_data.append(msh.cell_data["gmsh:physical"][i])
            else:
                line_data.append(np.zeros(len(cell_block.data)))

    if line_cells:
        line_mesh = meshio.Mesh(
            points=msh.points,
            cells=line_cells,
            cell_data={"borda_tag": line_data}
        )
        
        meshio.write("malha_bordas.xdmf", line_mesh)
        print("    Sucesso: 'malha_bordas.xdmf' criado.")
    else:
        print("    AVISO: Nenhuma linha de fronteira encontrada.")

import subprocess

if __name__ == "__main__":
    # NOME DOS ARQUIVOS
    ARQUIVO_GEO = "malha_tubular.geo"
    ARQUIVO_MSH = "malha_tubular.msh"

    # 1. Gerar .msh a partir do .geo usando Gmsh
    print(f"Gerando arquivo .msh com Gmsh a partir de {ARQUIVO_GEO}...")
    ret = subprocess.run(["gmsh", ARQUIVO_GEO, "-2", "-format", "msh2", "-o", ARQUIVO_MSH])
    if ret.returncode != 0:
        raise RuntimeError("Erro ao rodar Gmsh. Verifique se o Gmsh está instalado e no PATH.")
    print(f"Arquivo {ARQUIVO_MSH} gerado com sucesso.")

    # 2. Converter para XDMF
    converter_gmsh_para_xdmf(ARQUIVO_MSH)

    # 3. Resumo da malha
    print("\nResumo da malha gerada:")
    msh = meshio.read(ARQUIVO_MSH)
    print(f"  Nós: {msh.points.shape[0]}")
    for cell_block in msh.cells:
        print(f"  Elementos {cell_block.type}: {cell_block.data.shape[0]}")

    print("\nPipeline de conversão finalizado!")