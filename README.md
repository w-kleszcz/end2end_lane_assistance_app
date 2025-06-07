# end2end_lane_assistance_app

Repozytorium pracy dyplomowej "Implementacja prototypu planera end-to-end"

---
## Przygotowanie środowiska

1. Przejdź do głównego folderu repozytrium
2. Zainicjuj wirtualne środowisko venv (możesz użyć dowolnej innej nazwy):
   ```bash
   python -m venv venv
   ```
3. Uruchom wirtualne środowisko (zastąp venv nazwą twojego środowiska, jeśli użyłeś innej w kroku 2.):
   ```bash
   source venv/bin/activate
   ```
4. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt
   ```

---
## Uruchomienie aplikacji

```bash
python src/app.py
```
W celu uruchomienia MLFlow:
```bash
mlflow ui
```
Wyniki zapisane do MLFlow można zwizualizować w http://localhost:5000

---
## Obsługa danych

Kontrola wersji nie obejmuje danych używanych w aplikacji.

Obecnie obsługiwanym datasetem jest https://drive.google.com/file/d/1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B/view

Odpakuj pobrane archiwum w folderze data/raw

---
## Przygotowanie danych

Dla wygody użytkownika, do repozytorium dodano plik dataset_recipe.yaml, który zawiera metadane z obróbki wyżej wymienionego datasetu. Rekomendujemy użycie tego pliku do pracy z aplikacją w krokach trenowania i ewaluacji modelu.