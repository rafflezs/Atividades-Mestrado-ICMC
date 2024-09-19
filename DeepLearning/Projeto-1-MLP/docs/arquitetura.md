# Regressão vs. Classificação
Embora algumas vertentes de pesquisas em implementações de redes neurais comerciais prefiram a separação entre modelos para __regressão__ e __classificação__,
foi acordado entre o grupo o desenvolvimento de uma só arquitetura que realiza ambas as tarefas.

O que diferencia a aplicação é o número de neurônios na camada de saída, sendo __3__ neurônuios para a tarefa de classificação (1 neurônio para cada classe),
e __1__ neurônio para a tarefa de regressão, retornando apenas um valor que expressa a saída linear do valor previsto.

# Da implementação

## Inicialização

A rede é construída com 5 parâmetros obrigatórios informados pelo usuário:

- __n_input:__ Valor inteiro expressando a quantidade de neurônios na camada de entrada.
- __n_hidden:__ Vetor de valores inteiros expressando as quantidades de neurônios em cada camada oculta.
- __n_output:__ Valor inteiro expressando a quantidade de neurônios na camada de saída.
- __function_hidden:__ Função de ativação das camadas ocultas
- __function_output:__ Função de ativação da camada de saída

Dentro do processo de inicialização da classe são construídos os seguintes atributos internos:

- __Peso (weight):__ Cada camada possui sua própria matriz de pesos que são inicializados com valores aleatórios.
- __Viés (bias):__ O vetor de bias, inicializado com valores 0 para cada camada.

## Foward Pass
Esse método é responsável pelo fluxo de propagação interno da rede. A entrada flui por cada camada, onde neurônios computam a o __produto interno__
das entradas somadas aos seus viezes.

> Formulação matemática do produto interno: $$Z^{(l)} = W^_{l} X + b^_{l}$$, onde $X$ representa a entrada da camada atual, $W$ e $b$ representam
os pesos e viezes da camada, respectivamente.

### Função de Ativação
É importante que seja aplicada uma função de ativação ao final da computação da camada, responsável pela filtragem da saída da camada baseada em um
certo valor.
As funções adotadas foram:

- __ReLU:__
- __Sigmoide:__

## Backpropagation
O algoritmo de _backpropagation_ (ou retrocesso) é responsável pelo cálculo dos _deltas_ (erros), implementados no aprendizado para verificar a
distância do modelo ao valor real, realizando de maneira recursiva a atualização dos pesos e viezes do modelo de modo à calibra-lo para a próxima
iteração de aprendizado.
O algoritmo implementado realiza as seguintes operações:

1. __Loss:__ Computação da diferença entre o valor __predito__ e o valor __alvo__ através da diferença dos quadrados - $(y - t)^2$.
2. __Delta:__ O erro da camada de saída é computado através da multiplicação do _loss__ com a derivada da função de ativação.

$$\sig_{l} = (W_{l+1})^T \sig_{l+1} \dot f' (Z_{l})$$

### Atualização dos pesos e viezes
Com o cálculo dos erros através do _retrocesso_ do modelo, os valores de peso e viés de cada camada são recalculados para calibração do aprendizado
do modelo através do método de __descida do gradiente__:

$$W_{l} = W_{l} + \eta \sig_{l} X_{l-1}$$

## Fluxo de operações
### Treino
O treino é realizado ao longo de múltiplos ciclos (épocas), onde cada iteração realiza a sequência:

Foward Pass \rightarrow Backpropagation \rightarrow Atualização dos pesos

### Predição
A predição com o modelo treinado realiza apenas a operação de __Foward Pass__, retornando a probabilidade da ativação do neurônio de cada classe.