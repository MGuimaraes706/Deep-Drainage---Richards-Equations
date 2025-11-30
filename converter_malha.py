import meshio
import os

def converter_gmsh_para_xdmf(nome_arquivo_msh):
    """
    Converte um arquivo .msh (Gmsh) para dois arquivos .xdmf (Python/Paraview/FEniCS).
    Separa a geometria 2D (triângulos) das fronteiras 1D (linhas).
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
    # Procura por células do tipo 'triangle'
    triangle_cells = [c for c in msh.cells if c.type == "triangle"]
    
    if triangle_cells:
        # Extrai os dados físicos (Tags) associados aos triângulos
        # O meshio guarda isso em msh.cell_data['gmsh:physical']
        # Precisamos pegar o índice correto correspondente aos triângulos
        
        # Encontra o índice no array de células onde estão os triângulos
        index_tri = -1
        for i, c in enumerate(msh.cells):
            if c.type == "triangle":
                index_tri = i
                break
        
        if index_tri != -1:
            triangle_data = msh.cell_data["gmsh:physical"][index_tri]
            
            # Cria a nova malha limpa só de triângulos
            triangle_mesh = meshio.Mesh(
                points=msh.points,
                cells=[("triangle", triangle_cells[0].data)],
                cell_data={"material_tag": [triangle_data]}
            )
            
            meshio.write("malha_dominio.xdmf", triangle_mesh)
            print("--> Sucesso: 'malha_dominio.xdmf' criado (Contém os Solos, Brita, Concreto).")
        else:
            print("AVISO: Células triangulares detectadas mas sem dados físicos associados.")
    else:
        print("ERRO CRÍTICO: Nenhum triângulo encontrado! Você gerou a malha 2D no Gmsh?")

    # ===================================================
    # 2. EXTRAIR AS BORDAS (LINHAS)
    # ===================================================
    # Procura por células do tipo 'line'
    line_cells = [c for c in msh.cells if c.type == "line"]
    
    if line_cells:
        index_line = -1
        for i, c in enumerate(msh.cells):
            if c.type == "line":
                index_line = i
                break
                
        if index_line != -1:
            line_data = msh.cell_data["gmsh:physical"][index_line]
            
            line_mesh = meshio.Mesh(
                points=msh.points,
                cells=[("line", line_cells[0].data)],
                cell_data={"borda_tag": [line_data]}
            )
            
            meshio.write("malha_bordas.xdmf", line_mesh)
            print("--> Sucesso: 'malha_bordas.xdmf' criado (Contém Entrada, Saída, Rio, etc).")
    else:
        print("AVISO: Nenhuma linha de fronteira encontrada. Verifique se você definiu 'Physical Curve' no .geo.")

if __name__ == "__main__":
    # ==========================================
    # CONFIGURAÇÃO
    # ==========================================
    # Mude aqui se o nome do seu arquivo for diferente
    ARQUIVO_ENTRADA = "malha_tubular.msh" 
    
    converter_gmsh_para_xdmf(ARQUIVO_ENTRADA)
