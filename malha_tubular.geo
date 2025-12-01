SetFactory("OpenCASCADE");

// ==========================================
// 1. PARÂMETROS GERAIS E DE MALHA
// ==========================================
lc = 0.5;           // Malha Solo (Grossa)
lc_brita = 0.15;    // Malha Brita (Média)
lc_concreto = 0.1;  // Malha Concreto (Fina)

// Solo
L = 10.0;
H_total = 6.0;
H_camada_inf = 2.5; // Interface entre Solo Inferior (Areia) e Superior (Argila)

// Aduela (Concreto)
w_int = 2.0;
h_int = 1.5;
espessura = 0.20;
x_aduela = 4.0; // Posição X do canto inferior esquerdo da Aduela externa
y_aduela = 1.0; // Posição Y do canto inferior esquerdo da Aduela externa

// Envelope de Brita (Transição)
esp_brita = 0.30; // 30cm de brita ao redor da aduela

// ==========================================
// 2. CRIAÇÃO DAS FORMAS BRUTAS
// ==========================================

// 1. Domínio principal do Solo (um único retângulo grande)
Solo_Bruto = news; Rectangle(Solo_Bruto) = {0, 0, 0, L, H_total};

// 2. Envelope de Brita Bruto (inclui Brita, Concreto e Duto)
w_brita_total = w_int + 2espessura + 2esp_brita;
h_brita_total = h_int + 2espessura + 2esp_brita;
x_brita_inicio = x_aduela - esp_brita;
y_brita_inicio = y_aduela - esp_brita;

Brita_Shape = news; 
Rectangle(Brita_Shape) = {
x_brita_inicio,
y_brita_inicio,
0, 
    w_brita_total, // Largura total
    h_brita_total // Altura total
};

// 3. Aduela Externa (Concreto Bruto)
Aduela_Ext = news; Rectangle(Aduela_Ext) = {x_aduela, y_aduela, 0, w_int+2espessura, h_int+2espessura};
// 4. Aduela Interna (Buraco do Rio)
Aduela_Int = news; Rectangle(Aduela_Int) = {x_aduela+espessura, y_aduela+espessura, 0, w_int, h_int};

// ==========================================
// 3. OPERAÇÕES BOOLEANAS (FRAGMENTAÇÃO)
// ==========================================
// Usamos Fragment para gerar todas as interseções e limites.

Print("Executando Fragmentação (Todas as entidades) - Isso garante que os limites internos sejam criados.");
todas_formas[] = {
Surface{Solo_Bruto},
Surface{Brita_Shape},
Surface{Aduela_Ext},
Surface{Aduela_Int}
};

// O Fragmento aplica todas as intersecções possíveis.
out[] = BooleanFragment{ todas_formas[] ; }{ todas_formas[] ; };
// A operação 'out[]' contém todas as superfícies fragmentadas.

// ==========================================
// 4. CLASSIFICAÇÃO PELO CENTRO DE MASSA
// ==========================================
// Agora, encontramos o centro de massa de cada novo pedaço para classificá-lo.

// Limites (Para o centro de massa)
x_conc_min = x_aduela; x_conc_max = x_aduela + w_int + 2espessura;
y_conc_min = y_aduela; y_conc_max = y_aduela + h_int + 2espessura;

x_duto_min = x_aduela + espessura; x_duto_max = x_aduela + espessura + w_int;
y_duto_min = y_aduela + espessura; y_duto_max = y_aduela + espessura + h_int;

// Tags de destino
id_solo_inf[] = {};
id_solo_sup[] = {};
id_brita[] = {};
id_concreto[] = {};
id_duto[] = {};

// Iterar sobre as novas superfícies geradas pelo Fragment
For i In {0:Length(out[])-1}
    If(Type(out[i]) == "Surface")
        mass[] = CenterOfMass { Surface{out[i]} };
        cx = mass[0];
        cy = mass[1];

        // 1. DUCTO/RIO (Dentro dos limites internos)
        If( (cx > x_duto_min) && (cx < x_duto_max) && (cy > y_duto_min) && (cy < y_duto_max) )
            id_duto[] += {out[i]};
        // 2. CONCRETO (Dentro dos limites da Aduela Externa, mas fora do Duto)
        Else If( (cx > x_conc_min) && (cx < x_conc_max) && (cy > y_conc_min) && (cy < y_conc_max) )
            id_concreto[] += {out[i]};
        // 3. BRITA (Dentro dos limites da Brita Bruta, mas fora do Concreto)
        Else If( (cx > x_brita_inicio) && (cx < x_brita_inicio + w_brita_total) && (cy > y_brita_inicio) && (cy < y_brita_inicio + h_brita_total) )
            id_brita[] += {out[i]};
        // 4. SOLO (O restante)
        Else
            If (cy < H_camada_inf)
                id_solo_inf[] += {out[i]};
            Else
                id_solo_sup[] += {out[i]};
            EndIf
        EndIf
    EndIf
EndFor

// ==========================================
// 5. REFINAMENTO LOCAL
// ==========================================

// Refinamento do Envelope de Brita
PointsOfBrita[] = PointsOf{ Surface{id_brita[]} };
Characteristic Length{PointsOfBrita[]} = lc_brita;

// Refinamento do Concreto
PointsOfConcreto[] = PointsOf{ Surface{id_concreto[]} };
Characteristic Length{PointsOfConcreto[]} = lc_concreto;

// ==========================================
// 6. GRUPOS FÍSICOS (Tags para Solver)
// ==========================================

// --- Superfícies (Domínios) ---
Physical Surface("Solo_Inferior") = {id_solo_inf[]};
Physical Surface("Solo_Superior") = {id_solo_sup[]};
Physical Surface("Envelope_Brita") = {id_brita[]};
Physical Surface("Concreto") = {id_concreto[]};
// O duto agora é uma entidade física garantida
Physical Surface("Duto_Rio") = {id_duto[]};

// --- Bordas (Condições de Contorno) ---
// As bordas precisam ser redefinidas para o novo solo fragmentado.

// Tolerância para seleção de linhas
eps = 1e-3;

// ESQUERDA (x = 0)
Linhas_Esq[] = Curve In BoundingBox {-eps, -eps, -eps, eps, H_total+eps, 0};
Physical Curve("Esquerda_Entrada") = {Linhas_Esq[]};

// DIREITA (x = L)
Linhas_Dir[] = Curve In BoundingBox {L-eps, -eps, -eps, L+eps, H_total+eps, 0};
Physical Curve("Direita_Saida") = {Linhas_Dir[]};

// BASE (y = 0) - Impermeável
Linhas_Base[] = Curve In BoundingBox {-eps, -eps, -eps, L+eps, eps, 0};
Physical Curve("Base_Impermeavel") = {Linhas_Base[]};

// TOPO (y = H_total)
Linhas_Topo[] = Curve In BoundingBox {-eps, H_total-eps, -eps, L+eps, H_total+eps, 0};
Physical Curve("Topo_Terreno") = {Linhas_Topo[]};

// INTERNO (Onde o Rio passa) - Contorno da Aduela Interna
// CombinedBoundary no duto (id_duto) agora funciona porque o duto é um volume.
Physical Curve("Parede_Interna_Rio") = CombinedBoundary{ Surface{id_duto[]}; };

// --- Pontos de Monitoramento ---
// Piezômetro 1: Longe da aduela
P1 = newp; Point(P1) = {1.0, 4.0, 0, lc};
// Piezômetro 2: Perto da aduela
P2 = newp; Point(P2) = {3.0, 2.0, 0, lc};
// Piezômetro 3: Embaixo da aduela
P3 = newp; Point(P3) = {x_aduela + w_int/2, y_aduela - 0.5, 0, lc};

// Forçar esses pontos na malha do solo (assumindo Solo Superior)
// Note: O comando 'Point In Surface' é opcional e pode ser removido se causar erros.
// Recomenda-se adicionar apenas os pontos físicos, e o Gmsh deve lidar com eles.
Physical Point("Piezo_Longe") = {P1};
Physical Point("Piezo_Perto") = {P2};
Physical Point("Piezo_Fundo") = {P3};

// ==========================================
// 7. GERAÇÃO DA MALHA
// ==========================================
Mesh.Algorithm = 6; // Algoritmo 2D (MeshAdapt)
Mesh.CharacteristicLengthFactor = 1.0; // Usar lc definido

// Para gerar a malha automaticamente:
// Mesh 2;
// Save "malha_aduela_fragmentada.msh";