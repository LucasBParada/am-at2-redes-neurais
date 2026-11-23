# AM-AT2 – Redes Neurais: faixa de preço de carros usados no Reino Unido

Projeto de classificação multiclasse desenvolvido com Scikit-Learn, utilizando `ColumnTransformer` para o pré-processamento, `MLPClassifier` para a classificação e FastAPI para disponibilizar uma API REST de inferência.

## Objetivo do projeto

O objetivo é classificar carros usados em três faixas de preço com base em características do veículo, como marca, quilometragem, ano de registro, número de donos, combustível, carroceria, motor, câmbio, portas, assentos, classe de emissão e histórico de manutenção.

As classes utilizadas são:

- `Economico`: preço até £3.000;

- `Intermediario`: preço acima de £3.000 até £6.500;

- `Premium`: preço acima de £6.500.

O preço original é utilizado somente para criar a variável-alvo e não é enviado como atributo preditor ao modelo.

## Dataset

O dataset utilizado é público, tabular e contém anúncios de carros usados no Reino Unido.

Fonte oficial no Kaggle:

[Used Car Prices in UK — Kaggle](https://www.kaggle.com/datasets/muhammadawaistayyab/used-cars-prices-in-uk)

O arquivo utilizado pelo projeto está em:

```
data/dataset.csv
```

O dataset original possui 3.685 registros e combina variáveis numéricas e categóricas. Após a limpeza dos dados, remoção de registros fora dos limites e remoção de duplicidades completas, são utilizadas 2.849 linhas no treinamento e na avaliação.

## Resultado do modelo

A avaliação foi realizada com `train_test_split`, utilizando divisão estratificada e `random_state=77`.

Resultados registrados:

```
Acurácia: 0.8281
MCC: 0.7390
```

O MCC, ou Matthews Correlation Coefficient, foi utilizado junto com a acurácia para oferecer uma avaliação mais completa do desempenho multiclasse.

O campo `score_confianca` retornado pela API corresponde à probabilidade bruta produzida por `predict_proba`. Esse valor não representa uma probabilidade calibrada.

## Estrutura do projeto

```
am-at2-redes-neurais/
├── data/
│   └── dataset.csv
├── models/
│   └── pipeline_rede_neural.joblib
├── notebooks/
│   └── analise_treinamento.ipynb
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── features.py
│   ├── schemas.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_features.py
├── .gitignore
├── pyproject.toml
├── requisicoes.http
├── README.md
└── uv.lock
```

## Requisitos

- Python entre as versões 3.11 e 3.13;

- `uv` instalado;

- acesso à internet para instalar as dependências na primeira execução.

O ambiente virtual `.venv` não deve ser incluído no repositório ou no arquivo ZIP. Ele é criado automaticamente pelo `uv`.

## Instalação

Abra o terminal na raiz do projeto e execute:

```bash
uv sync --group dev
```

Esse comando instala as dependências de produção e de desenvolvimento usando o `uv.lock`.

## Treinamento do modelo

O treinamento reproduzível está disponível no script `src/train.py`.

Para treinar novamente e gerar o pipeline serializado:

```bash
uv run python -m src.train
```

O modelo será salvo obrigatoriamente em:

```
models/pipeline_rede_neural.joblib
```

Também é possível informar caminhos alternativos:

```bash
uv run python -m src.train \
  --data data/dataset.csv \
  --model models/pipeline_rede_neural.joblib
```

O pipeline salvo contém tanto o pré-processamento quanto o `MLPClassifier`. Dessa forma, as regras de imputação, padronização e codificação categórica utilizadas durante o treinamento são preservadas para a inferência.

## Notebook

O notebook contém:

- contexto do problema;

- origem e dicionário dos dados;

- análise da dimensionalidade do dataset;

- análise de valores ausentes;

- gráficos exploratórios;

- limpeza e engenharia de atributos;

- separação entre treino e teste;

- `ColumnTransformer`;

- `StandardScaler`;

- `OneHotEncoder`;

- `MLPClassifier`;

- acurácia;

- MCC;

- relatório de classificação;

- matriz de confusão;

- exportação do pipeline com `joblib`.

Para abrir o notebook:

```bash
uv run jupyter lab notebooks/analise_treinamento.ipynb
```

O notebook utiliza caminhos relativos ao projeto e não depende de um caminho específico do Windows.

O script `src/train.py` é a forma recomendada para reproduzir o treinamento automaticamente.

## Testes automatizados

O projeto possui testes automatizados para a API e para a engenharia de atributos.

Execute:

```bash
uv run pytest -q
```

Resultado esperado:

```
7 passed
```

A configuração abaixo, presente no `pyproject.toml`, garante que a pasta `src` seja encontrada pelo `pytest`, inclusive no Windows:

```
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-ra"
```

## Execução da API

Para iniciar o servidor:

```bash
uv run uvicorn src.app:app --reload
```

A API ficará disponível em:

```
http://127.0.0.1:8000
```

A documentação interativa do FastAPI está disponível em:

```
http://127.0.0.1:8000/docs
```

O pipeline é carregado uma única vez durante o evento de inicialização da aplicação, por meio do `lifespan`. O arquivo do modelo não é lido dentro das rotas.

## Endpoints

### Diagnóstico do modelo

```
GET /diagnostico/modelo
```

Resposta obrigatória conforme o enunciado:

```json
{
  "status": "operacional_v2",
  "framework": "FastAPI-Production",
  "modelo": "MLPClassifier"
}
```

### Predição

```
POST /predict
```

Exemplo de requisição:

```json
{
  "marca": "Ford",
  "quilometragem": 80000,
  "ano_registro": 2012,
  "numero_donos": 2,
  "combustivel": "Petrol",
  "carroceria": "Hatchback",
  "motor_litros": 1.6,
  "cambio": "Manual",
  "portas": 5,
  "assentos": 5,
  "classe_emissao": "Euro 5",
  "historico_completo": 0
}
```

Exemplo de resposta:

```json
{
  "categoria_prevista": "Intermediario",
  "score_confianca": 0.9198057575454989
}
```

O campo `numero_donos` é opcional. Quando não informado, o pipeline realiza a imputação configurada durante o treinamento.

## Validação das entradas

O esquema Pydantic rejeita entradas inválidas com HTTP `422 Unprocessable Entity`.

Entre as validações implementadas estão:

- quilometragem entre 0 e 300.000 milhas;

- ano de registro entre 1990 e 2024;

- motor entre 0,6 e 7,0 litros;

- portas entre 2 e 5;

- assentos entre 2 e 9;

- combustível pertencente às categorias aceitas;

- carroceria pertencente às categorias aceitas;

- câmbio `Manual` ou `Automatic`;

- classe de emissão entre `Euro 1` e `Euro 6`;

- histórico completo igual a 0 ou 1.

## Testes de integração HTTP

O arquivo `requisicoes.http` contém chamadas para testar a API usando a extensão REST Client do VS Code ou ferramentas compatíveis.

Todos os testes enviam obrigatoriamente:

```
User-Agent: Tester-Audit-Agent
```

O arquivo contém:

- uma chamada GET para `/diagnostico/modelo`;

- três chamadas válidas para `/predict`;

- uma chamada com texto em campo numérico;

- uma chamada com valores fora dos limites e categoria inexistente.

As requisições inválidas devem retornar HTTP `422`.

## Comandos principais

Instalar dependências:

```bash
uv sync --group dev
```

Executar testes:

```bash
uv run pytest -q
```

Treinar o modelo:

```bash
uv run python -m src.train
```

Iniciar a API:

```bash
uv run uvicorn src.app:app --reload
```

Abrir o notebook:

```bash
uv run jupyter lab notebooks/analise_treinamento.ipynb
```

## Observação sobre versões

O notebook, o script de treinamento e a API devem ser executados no ambiente criado pelo `uv`. O arquivo `uv.lock` mantém versões compatíveis das dependências, especialmente do Scikit-Learn, que é utilizado para carregar o arquivo `.joblib`.