from argparse import ArgumentParser

def tabela_verdade_xor(size = 2) -> list:  
    tabela_xor = []
    for i in range(2**size):
        valores_binarios = [int(x) for x in list(bin(i)[2:].zfill(size))]
        valor_xor = 0
        for j in range(len(valores_binarios) - 1):
            valor_xor ^= bool(valores_binarios[j]) != bool(valores_binarios[j + 1])

        tabela_xor.append(','.join(map(str, valores_binarios + [int(valor_xor)])))
    
    return tabela_xor

def gravar_xor_arquivo(tam_tabela: int, tabela_verdade: list) -> None:
    with open(f"data/XOR/n{tam_tabela}.txt", "w") as file:
        for linha in tabela_verdade:
            file.write(f"{linha}\n")

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-n", "--size", type=int, default=2, help="Tamanho da tabela verdade")
    args = parser.parse_args()
    
    gravar_xor_arquivo(args.size, tabela_verdade_xor(args.size))
    print("Tabela verdade XOR gerada com sucesso!")