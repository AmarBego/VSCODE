1. Datengrundlage
    - Beschreibung der Spalten:
        - gender: Kategorie, Geschlecht der Prüflinge.
        - race/ethnicity: Kategorie, Rasse/Ethnizität der Prüflinge.
        - parental level of education: Kategorie, Bildungsniveau der Eltern.
        - lunch: Kategorie, Art des Mittagessens. 
        - test preparation course: Kategorie, gibt es ein Vorbereitungskurs.
        - math score: Numerisch, Mathematik-Ergebnis.
        - reading score: Numerisch, Leseverständnis-Ergebnis.
        - writing score: Numerisch, Schreibfähigkeit-Ergebnis.

    - Vollständige EDA (Explorative Datenanalyse):
        - Grafische Darstellung der Verteilungen jeder Variablen.
        - Analyse fehlender Werte.
        - Untersuchung der Beziehungen zwischen den Kategorien und den Leistungen.

2. Hypothesen
    - **Beispiele für Hypothesen:**
        1. Schüler mit Vorbereitungskurs erzielen höhere Testergebnisse.
        2. Das Bildungsniveau der Eltern beeinflusst die Leistung der Schüler.
        3. Es gibt einen signifikanten Unterschied in den Leistungen zwischen den verschiedenen Ethnien.

    - **Überprüfung der Hypothesen:**
        - Einsatz statistischer Tests (z.B. t-Test, Chi-Quadrat-Test) zur Überprüfung der Hypothesen.

3. **Vorgehen Modellierung (40%)**
    - **Untersuchung von Variablen-Paaren/Gruppen:**
        - Korrelationsanalysen.
        - Visualisierung von Beziehungen.
    - **Modellerstellung:**
        - Mindestens zwei Modelle, z.B. lineare Regression, Entscheidungsbaum.
    - **Performance-Analyse:**
        - Bewertung der Modelle anhand von Metriken wie R², MSE (Mean Squared Error).
        - Vergleich der Modelle.

4. **Summary (20%)**
    - **Zusammenfassung der Ergebnisse:**
        - Haupterkenntnisse aus der EDA und Modellierung.
    - **Diskussion ethischer Probleme:**
        - Berücksichtigung von Datenschutz, Bias in den Daten.

5. **Formales (10%)**
    - **Gestaltung der Dokumentation:**
        - Klare Struktur, gute Lesbarkeit.
        - Einsatz von Grafiken zur Veranschaulichung.
        - Achten auf Rechtschreibung und Stil.

Für die Codierung und Analyse können Sie Python mit Bibliotheken wie Pandas, NumPy, Matplotlib, Seaborn und Scikit-learn verwenden. In Ihrem Jupyter-Notebook sollten alle relevanten Codeabschnitte und Visualisierungen enthalten sein, die Ihre Aussagen und Ergebnisse stützen.

Für eine umfassende Bearbeitung Ihres Projektes mit dem Datensatz "Exams.csv" im Rahmen des CRISP-DM-Prozessmodells, würde der Prozess wie folgt aussehen:

### 1. Datengrundlage (20%)
Zuerst wird eine Explorative Datenanalyse (EDA) durchgeführt, um einen Überblick über den Datensatz zu bekommen.

#### EDA-Code:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Daten einlesen
df = pd.read_csv('Exams.csv')

# Grundlegende Datenüberprüfung
print(df.head())
print(df.describe())
print(df.info())

# Verteilung der numerischen Variablen
for column in ['math score', 'reading score', 'writing score']:
    plt.figure(figsize=(10,4))
    sns.histplot(df[column], kde=True)
    plt.title(f'Verteilung von {column}')
    plt.show()

# Überprüfung auf fehlende Werte
print(df.isnull().sum())

# Verteilung der kategorischen Variablen
for column in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
    plt.figure(figsize=(10,4))
    sns.countplot(x=column, data=df)
    plt.title(f'Verteilung von {column}')
    plt.xticks(rotation=45)
    plt.show()
```

### 2. Hypothesen (10%)
Hypothesen basierend auf der EDA:

1. **Hypothese:** Schüler, die den Vorbereitungskurs absolviert haben, erzielen im Durchschnitt höhere Testergebnisse.
2. **Hypothese:** Es gibt einen signifikanten Zusammenhang zwischen dem Bildungsniveau der Eltern und den Testergebnissen der Schüler.
3. **Hypothese:** Mädchen und Jungen unterscheiden sich signifikant in mindestens einem der Testergebnisse.

#### Hypothesen-Test-Code:
```python
from scipy import stats

# Hypothese 1
prep_course_group = df.groupby('test preparation course')['math score', 'reading score', 'writing score'].mean()
print(prep_course_group)

# Hypothese 2
parent_edu_group = df.groupby('parental level of education')['math score', 'reading score', 'writing score'].mean()
print(parent_edu_group)

# Hypothese 3 - T-Test für unabhängige Stichproben
for score in ['math score', 'reading score', 'writing score']:
    t_stat, p_val = stats.ttest_ind(df[df['gender'] == 'female'][score],
                                    df[df['gender'] == 'male'][score])
    print(f"{score} - T-Statistik: {t_stat}, P-Wert: {p_val}")
```

### 3. Vorgehen Modellierung (40%)
Hier werden zwei Modelle erstellt, um die Zusammenhänge im Datensatz zu untersuchen und zu prognostizieren.

#### Modellierungs-Code:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Modell für Mathematik-Ergebnis
X = pd.get_dummies(df.drop(['math score', 'reading score', 'writing score'], axis=1))
y = df['math score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'MSE für Mathematik-Modell: {mean_squared_error(y_test, y_pred)}')
```

### 4. Summary (20%)
Eine Zusammenfassung der Resultate und eine Diskussion über ethische Probleme der Modelle.

### 5. Formales (10%)
Die Dokumentation sollte übersichtlich gestaltet und gut strukturiert sein, unter Verwendung von Grafiken und Tabellen, um die Ergebnisse zu visualisieren und den Lesefluss zu fördern.

Sie müssten den tatsächlichen Code in einem Jupyter-Notebook ausführen und die Ergebnisse für jeden Schritt dokumentieren. Für eine umfassende Analyse wären zusätzliche Details und eine feinere Anpassung der Modelle erforderlich.

