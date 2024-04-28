import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def ler_instancia_gurobipandas(arquivo):
    return pd.read_csv(arquivo)

def resolver_modelo(instancia_ingredientes, instancia_demanda, instancia_epsilon):
    model = gp.Model("Mistura-Relaxada")
    
    ingredientes = instancia_ingredientes.set_index('Ingrediente')
    demanda = instancia_demanda.set_index('Mistura')
    epsilon = instancia_epsilon.set_index('Mistura')

    # Indices
    N_ingredientes = len(ingredientes)
    M_chas = len(demanda)
    K_caracteristicas = len(ingredientes.columns) - 2
    
    # Parâmetros
    Pi_custo =  ingredientes['Preço (£0.01/kg)']
    D_j_demanda = demanda['Demanda']
    a_i_disponibilidade = ingredientes['Disponibilidade (kg)']
    g_ik_caracteristica_ingrediente = ingredientes[['Caracateristica1','Caracateristica2','Caracateristica3','Caracateristica4','Caracateristica5','Caracateristica6']]
    s_jk_pontuacao_alvo_caracteristica = demanda[['AlvoCaracterisca1','AlvoCaracterisca2','AlvoCaracterisca3','AlvoCaracterisca4','AlvoCaracterisca5','AlvoCaracterisca6']]

    # Variáveis de decisão
    x = model.addVars(N_ingredientes, M_chas, name="x")

    # FO
    model.setObjective(gp.quicksum((Pi_custo.iloc[i] * x[i, j]) for i in range(1,N_ingredientes) for j in range(1,M_chas)), GRB.MINIMIZE)

    # R1 - disponibilidade de matéria-prima
    for i in range(N_ingredientes):
        model.addConstr(gp.quicksum(x[i, j] for j in range(M_chas)) <= a_i_disponibilidade.iloc[i])

    # R2 - demanda das misturas
    for j in range(M_chas):
        model.addConstr(gp.quicksum(x[i, j] for i in range(N_ingredientes)) >= D_j_demanda.iloc[j])

    # R3 e R4 - pontuação das características (relaxadas)
    for j in range(M_chas):
        for k in range(K_caracteristicas):
            model.addConstr(gp.quicksum((g_ik_caracteristica_ingrediente.iloc[i,k] - s_jk_pontuacao_alvo_caracteristica.iloc[j,k] - epsilon.iloc[j,k]) * x[i, j] for i in range(N_ingredientes)) <= 0)
            model.addConstr(gp.quicksum((g_ik_caracteristica_ingrediente.iloc[i,k] - s_jk_pontuacao_alvo_caracteristica.iloc[j,k] + epsilon.iloc[j,k]) * x[i, j] for i in range(N_ingredientes)) >= 0)

    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("Solução ótima:")
        for i in range(N_ingredientes):
            for j in range(M_chas):
                print(f"x[{i},{j}] = {x[i, j].x}")
    print("Valor ótimo da função objetivo:", model.objVal)

def main():
    instancia_ingredientes = ler_instancia_gurobipandas('ingredientes.csv')
    instancia_demanda = ler_instancia_gurobipandas('demanda.csv')
    epsilon = ler_instancia_gurobipandas('epsilon.csv')

    resolver_modelo(instancia_ingredientes, instancia_demanda, epsilon)

if __name__ == '__main__':
    main()