# Desafio Indicium - Previsão de Nota IMDB

## Instalação e execução

### 1. Clone o repositório
```bash
git clone https://github.com/not-felipe/LH_CD_FELIPEALVES
cd LH_CD_FELIPEALVES
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute os notebooks
Inicie o Jupyter Notebook:
```bash
jupyter notebook
```

Abra os seguintes arquivos:

1. `EDA_imdb.ipynb` (este faz o EDA e pré-processamento do dataset)
2. `Modelo.ipynb` (este aplica transformações nas features e implementa o modelo)

### 4. Utilização do pkl
Para utilizar o pkl será necessário criar alguns features e pré processar os dados. Para facilitar esse processo, criei a pasta src, que contém o arquivo aux.py.
Abaixo, colocarei um exemplo de utilização:
```python
import pandas as pd
import pickle
from src.aux import preprocess_data, create_features

df = pd.read_csv("data/desafio_indicium_imdb.csv")

df = preprocess_data(df)

df_model = create_features(df)

# Carregar modelo treinado
with open("models/imdb_rating_model.pkl", "rb") as f:
    model = pickle.load(f)

new_film = {
    'Series_Title': 'The Shawshank Redemption',
    'Released_Year': 1994,
    'Certificate': 'A',
    'Runtime': '142 min',
    'Genre': 'Drama',
    'Meta_score': 80.0,
    'Director': 'Frank Darabont',
    'Star1': 'Tim Robbins',
    'Star2': 'Morgan Freeman',
    'Star3': 'Bob Gunton',
    'Star4': 'William Sadler',
    'No_of_Votes': 2343110,
    'Gross': '28,341,469'
}
new_df = pd.DataFrame([new_film])

# Aplicar mesmo pré-processamento e features
new_df = preprocess_data(new_df, df_original=df)
new_df = create_features(new_df)

X_new = new_df[[
    "Meta_score", "Runtime", "No_of_Votes",
    "Primary_Genre", "Certificate",
    "Director_frequency", "Star1_frequency"
]]

# Aplicar encoders salvos
with open("models/label_encoders.pkl", "rb") as f:
    le_dict = pickle.load(f)

for col in ["Primary_Genre", "Certificate"]:
    X_new[col] = le_dict[col].transform(X_new[col].astype(str))

print("Previsão IMDB:", model.predict(X_new)[0])
