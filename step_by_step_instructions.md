Krok 1: Načítanie a príprava dát
Načítaj tréningové dáta pomocou numpy.load(). K dispozícii sú súbory X_public.npy a y_public.npy. Tieto súbory obsahujú príznaky (X_public) a zodpovedajúce odpovede (y_public).

Použi tento kód:
python
Copy code
import numpy as np

X_public = np.load('X_public.npy')
y_public = np.load('y_public.npy')
Načítaj evaluačné dáta (X_eval), na ktorých budeš testovať svoj model:

Použi tento kód:
python
Copy code
X_eval = np.load('X_eval.npy')
Krok 2: Výber a tréning regresného modelu
Vyber vhodný regresný model. Môžeš použiť modely zo scikit-learn, ako napríklad lineárnu regresiu, ridge regresiu, random forest, alebo iné regresné modely.

Natrénuj model na dátach X_public a y_public. Ak použiješ napríklad lineárnu regresiu, kód bude vyzerať takto:

Použi tento kód:
python
Copy code
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_public, y_public)
Vyskúšaj rôzne modely (ak máš čas), a rozhodni sa, ktorý funguje najlepšie na tvojich dátach. Môžeš testovať rôzne regresné modely a porovnať ich výsledky na tréningových dátach pomocou metriky R².

Krok 3: Predikcia pre nové dáta (X_eval)
Použi natrénovaný model na predikciu hodnôt pre evaluačné dáta X_eval. Toto sú dáta, na ktorých bude tvoj model hodnotený:

Použi tento kód:
python
Copy code
y_eval = model.predict(X_eval)
Ulož predikované hodnoty do súboru y_predikcia.npy pomocou príkazu numpy.save():

Použi tento kód:
python
Copy code
np.save('y_predikcia.npy', y_eval)
Krok 4: Vyhodnotenie modelu
Vyhodnoť presnosť tvojho modelu na tréningových dátach (X_public a y_public) pomocou metriky R² (r2_score). Týmto zistíš, ako dobre tvoj model funguje na dátach, na ktorých bol trénovaný:
Použi tento kód:
python
Copy code
from sklearn.metrics import r2_score

y_train_pred = model.predict(X_public)
r2 = r2_score(y_public, y_train_pred)
print(f'R² na tréningových dátach: {r2}')
Krok 5: Vytvorenie technickej správy
Napíš stručný teoretický popis modelu: Popíš základné teoretické princípy modelu, ktorý si použil. Napríklad, ak si použil lineárnu regresiu, vysvetli, ako funguje lineárna regresia a prečo si ju vybral.

Vysvetli metodológiu riešenia: Opíš, ako si postupoval pri výbere modelu, aké metódy si použil a prečo. Napríklad, ak si skúšal viaceré modely, zdôvodni, prečo si vybral ten konkrétny model.

Diskusia o výsledkoch: Zahrň výsledky, ktoré si dosiahol, porovnaj rôzne prístupy, ktoré si skúšal, a diskutuj o tom, aké boli tvoje závery. Uveď, aké hodnoty R² si dosiahol na tréningových dátach, prípadne, čo by si ešte vylepšil.

