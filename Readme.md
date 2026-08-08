# ManaML: Magic: The Gathering Mana Value Predictor

In diesem Projekt habe ich ein ML-Modell zur Vorhersage des Mana Values (Converted Mana Cost / CMC) von MTG-Karten trainiert. Das Modell nutzt den Kartentext, Kampfwerte und Eigenschaften von Karten, um den Mana Value zu schätzen und Vorhersagen mit den tatsächlichen Kosten zu vergleichen. Vorrangig ist dieser Code darauf ausgelegt, die Hobbit-Karten zu überprüfen, kann aber durch Veränderung der Trainings-/Testdaten in der `data_loader.py` angepasst werden.

---

## Projektstruktur

```text
ManaML/
├── data/
│   ├── all-cards.jsonl     # Scryfall Bulk-Datensatz (nicht im Repo enthalten)
│   └── processed/          # Gecachte Vektoren und aufbereitetes DataFrame
├── models/                 # Gespeicherte Modell-Artefakte (.pkl)
├── src/
│   ├── data_loader.py      # Parsen und Säubern der Scryfall JSONL-Dateien
│   ├── features.py         # TF-IDF-Vektorisierung und Feature-Bereitstellung
│   ├── evaluator.py        # Metrik-Berechnung (MAE, RMSE, R²) und Modellspeicherung
│   └── analyzer.py         # Feature Importance und Ausreißer-Analyse
├── app.py                  # Streamlit Web-Interface (KI-generiert)
├── main.py                 # Hauptskript zum Trainieren und Evaluieren
└── requirements.txt        # Python-Abhängigkeiten
```

## Eingabevariablen (Features) und Datenaufbereitung

Der Trainingsdatensatz umfasst ca. 36.000 MTG-Karten aus der Scryfall-Datenbank. Als ungesehener Testdatensatz dienen 203 Karten aus den Hobbit-Sets (HOB / HOC).

### Extrahierte Eingabefeatures

**Textmerkmale (NLP):**
*   **Oracle Text:** Entfernung von Regel-Erklärungen in Klammern (Reminder Text), um Rauschen zu reduzieren.
*   **Multi-Face Cards:** Bei Adventure- und Split-Karten wird der Regeltext aus den Unterkarten (`card_faces`) zusammengefügt.
*   **TF-IDF N-Grams (1–3):** Extraktion der 2.000 aussagekräftigsten Begriffe und Phrasen. Seltene Tokens (`min_df=4`) sowie platzhaltende Meta-Wörter werden herausgefiltert.

**Numerische & Kategoriale Features:**
*   **Kampfwerte:** `power` und `toughness`.
*   **Farbkomplexität:** `colors_count` (Anzahl der im Manatyp vertretenen Farben).
*   **Textstatistiken:** `text_length` (Länge des gesäuberten Regeltextes) sowie `max_number_in_text` (höchste extrahierte Zahl im Text bis maximal 20).
*   **Typen-Flags:** Binäre Indikatoren für `is_creature`, `is_spell`, `is_artifact` und `is_enchantment`.

### Zielvariablen-Transformation

Die Zielvariable `cmc` wird vor dem Training über `log1p` ($\log(1 + \text{CMC})$) logarithmiert, um extreme Abweichungen bei teuren Sprüchen zu dämpfen. Bei der Auswertung wird das Ergebnis mit `expm1` zurückgerechnet.

---

## Modellanalyse und Ergebnisse

Es werden zwei Regressionsmodelle trainiert und verglichen:

*   **Hist Gradient Boosting Regressor:** $R^2 \approx 0,62$ | $\text{MAE} \approx 0,73$ Mana
*   **Random Forest Regressor:** $R^2 \approx 0,57$ | $\text{MAE} \approx 0,77$ Mana

### Wichtigste Feature Importances

Die Analyse über das Modul `src/analyzer.py` zeigt, welche Merkmale die stärkste Auswirkung auf die Manakosten-Vorhersage haben:

*   **Kampfwerte:** `power` und `toughness` dominieren die Baum-Entscheidungen bei Kreaturen (Angriffsstärke ist dabei ca. dreimal so stark gewichtet wie Widerstandskraft).
*   **Textumfang:** `text_length` und `max_number_in_text` dienen als starke Indikatoren für die Komplexität und Skalierung eines Effekts.
*   **Schlüsselwörter und Mechaniken:** Begriffe wie *target*, *targets*, *flying*, *control* und `is_artifact` bilden die wichtigsten regelbezogenen Signale.

---

## Setup und Installation

### 1. Umgebung einrichten und Bibliotheken installieren

Das Projekt kann wahlweise mit Conda oder pip (venv) aufgesetzt werden:

**Option A: Mit Conda**
```bash
# Umgebung erstellen und aktivieren
conda create -n manaml python=3.10 -y
conda activate manaml

# Pakete über pip in die Conda-Umgebung installieren
pip install -r requirements.txt
```

**Option B: Mit venv**
```bash
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Daten beschaffen (Scryfall Bulk Data)

Der Datensatz muss vor dem ersten Ausführen von Scryfall heruntergeladen werden:

1. Offizielle [Scryfall Bulk Data](https://scryfall.com/docs/api/bulk-data) Seite besuchen.
2. Lade dort den Datensatz **All Cards** im JSONL-Format herunter.
3. Platziere die heruntergeladene Datei im Ordner `data/` und benenne sie in `all-cards.jsonl` um:
    ```text
    data/all-cards.jsonl
    ```

---

## Ausführung

### Erstmaliger Durchlauf
Berechnet TF-IDF und speichert den Cache in `data/processed/`:
```bash
python main.py
```

### Schneller Folgedurchlauf
Lädt fertige Numpy-Arrays und DataFrames aus dem Cache:
```bash
python main.py --reuse-features
```

### Web-Interface
Hier können Karten aus der Datenbank ausgewählt werden und der Vergleich zwischen realem und vorhergesagtem Mana Value in einer UI für Einzelkarten durchgeführt werden.
```bash
streamlit run app.py
```

> **Hinweis zur Entwicklung:** Da es mir für dieses Projekt um den ML-Teil und nicht die UI ging, ist die Benutzeroberfläche (`app.py`) durch KI erstellt worden.