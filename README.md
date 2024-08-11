# Atelier Titanic
## Support de cours
<a href="https://docs.google.com/presentation/d/14GMhonuvL_6bNaMRWkwQCK9430jVEHKv2diNoC4tM_Y/edit?usp=sharing" target="_blank">Lien vers le support</a>

## Recupération des bibliothèques et du Jeu de données

### Recupération des bibliothèques
```
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers
```

### Recupération, netoyage et foramatage  du Jeu de données
```
titanic = sns.load_dataset("titanic")
titanic = titanic.dropna(subset=["survived", "sex", "pclass", "age", "fare"])
titanic["sex_num"] = titanic["sex"].map({"male": 0, "female": 1})
titanic = titanic.reset_index()
titanic = titanic[["sex_num", "pclass", "age", "fare", "survived"]]
titanic
```

## Découpage du tableau pour le faire passer dans le modèle

### Sélection des colonnes à garder pour faire la prédiction
```
X = titanic[["sex_num", "pclass", "age", "fare"]]
X
```

### La colonne à prédire
```
y = titanic["survived"]
y
```

## Création et entrainement du modèle

### Création du modèle
```
model = Sequential()
model.add(layers.Dense(10, activation='relu', input_dim=4)) 
model.add(layers.Dense(5, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
```

### Entrainement du modèle
```
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
model.fit(X, y, batch_size=16, epochs=20)
```

### Quelle serait la ligne qui me correspond
```
my_X = pd.DataFrame({"sex_num": [0], "pclass": [1],	"age": [39], "fare": [50.0]})
my_X
```

## Est ce que j'aurais survécu?
```
result = model.predict(my_X)[0,0] * 100
f"Vous avez %.2f%% de chances de survivre au titanic" % result
```
