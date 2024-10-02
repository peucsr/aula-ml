import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  # Importa o módulo pyplot de Matplotlib para criar gráficos
from sklearn.model_selection import train_test_split  # Função para dividir dados em treino e teste
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  # Função para calcular o Erro Quadrático Médio (MSE)


# dados fictícios
np.random.seed(0)
num_samples = 100

# dados de entrada 
idade = np.random.randint(20, 80, size=(num_samples, 1))  # Idade entre 20 e 80 anos
peso = 50 + 30 * np.random.rand(num_samples, 1)  # Peso entre 50 e 80 kg
condicoes_saude = np.random.randint(0, 2, size=(num_samples, 1))  # 0 = sem condição, 1 = com condição

# custo do tratamento (y)
# regra: custo base + impacto da idade, peso e condições de saúde
custo = 5000 + (idade * 50) + (peso * 30) + (condicoes_saude * 1000) + np.random.randn(num_samples, 1) * 100

# DataFrame para organizar os dados
dados = np.hstack([idade, peso, condicoes_saude])
df = pd.DataFrame(dados, columns=['idade', 'peso', 'condicoes_saude'])
df['custo'] = custo

# separação dos dados de treino e teste
X = df[['idade', 'peso', 'condicoes_saude']]
y = df['custo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# previsões
y_pred = model.predict(X_test)

# avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Erro quadrático médio (MSE): {mse}')

# visualizando a comparação entre os valores reais e os previstos
plt.scatter(y_test, y_pred, color='blue', label='Valores previstos vs reais')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Linha perfeita')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Hospital da Bahia')
plt.legend()
plt.show()
