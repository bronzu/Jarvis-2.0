# Pre-registrazione — Protezione per-frase vs protezione aggregata: geometria al primo ordine

Esperimento a ipotesi pre-registrata; non ricerca della soluzione.

## DEFINIZIONI

- d: numero di parametri addestrabili della configurazione.
- Insiemi disgiunti (quelli di Missione 1): X_forget; R_prot (frasi protette, che
  costruiscono i vincoli; parti da X_retain_proj); E_corr (le 12 correlate, NON
  protette); E_canary; E_ood (mai viste).
- Gradiente di una frase s: g_s = gradiente della NLL media per token di s rispetto ai
  parametri addestrabili. J = matrice le cui righe sono le g_s, s in R_prot, ciascuna
  normalizzata a norma 1 (N x d).
- P_k = proiettore sullo span dei primi k vettori singolari destri di J (SVD sottile).
  k_full = rango numerico di J con tolleranza relativa rcond=1e-6 sul valore singolare
  massimo (sensibilità: 1e-3 e 1e-9). Griglia k = {1,2,4,8,16,...,k_full}.
  "agg1" = proiezione su UNA sola direzione, il gradiente aggregato di R_prot (la
  protezione di Missione 1 in versione euclidea). Verifica leggendo unlearning.py che
  Missione 1 usi davvero un solo vettore (u = F^-1 g_r) e riportalo nel report.
- g_f = gradiente della NLL media di X_forget. v_k = (I - P_k) g_f; versore
  v^_k = v_k / ||v_k||.
- rho(k) = ||v_k|| / ||g_f|| (frazione del segnale di forgetting che sopravvive).
  Riferimento analitico: rho_rand(k) = sqrt((d - k)/d).
- Passo: theta' = theta + s * v^_k, con s = alfa * ||theta||, alfa in
  {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}. NLL misurate con forward pass reale.
- Delta_f = 0.5 * (media NLL su E_ood - NLL di X_forget prima del passo), per stato. Se
  Delta_f < 1 nat/token lo stato è "non applicabile" (frase non memorizzata).
- s*(k): il più piccolo s della griglia per cui la NLL di X_forget cresce di almeno
  Delta_f. Se nessun s ci arriva: "non raggiunto" (dato censurato: si riporta, non si
  scarta).
- M(k): massimo su E_corr della variazione relativa assoluta della NLL media per frase,
  misurata a s*(k). Riporta anche quante frasi superano il 5% e le variazioni su
  E_canary e R_prot.
- Predizione al primo ordine per la frase e: pred_e = s * (g_e . v^_k). Errore del
  primo ordine sul forget a s*(k): |pred - misurato| / |misurato|.
- Le metriche differiscono da Missione 1 (NLL relativa, direzione fissa e non
  iterativa): non confrontare i numeri direttamente.

## STATI E GRIGLIA

- Parametrizzazioni: LoRA rank 4 e full-parameter (rank 8/16 solo se già supportati
  senza modifiche al codice).
- Seed 42, 43, 44 (Missione 1 usava solo 42), stessi iperparametri di Missione 1.
- Checkpoint per ogni run: 25%, 50%, 100% delle epoche di Missione 1.
- Stati PRIMARI: checkpoint 100% x 2 parametrizzazioni x 3 seed = 6 stati. Tutto il
  resto (25%, 50%, sweep della dimensione di R_prot a N/4, N/2, N) è ESPLORATIVO e va
  etichettato così.
- Salva i checkpoint in results/checkpoints/ (non tracciato) e riusali.

## IPOTESI (fissate ora)

H1 - La protezione per-frase generalizza alle correlate non protette:
     M(k_full) <= M(agg1) / 3 in almeno 5 dei 6 stati primari (stati con dato censurato
     riportati a parte, non contati né pro né contro).
H2 - Proteggere non uccide il segnale di forgetting: rho(k_full) >= 0.5 in almeno 5
     dei 6 stati primari.
H3 - Il primo ordine è valido al passo necessario per dimenticare: errore del primo
     ordine sul forget a s*(k_full) <= 25% in almeno 5 dei 6 stati primari.
Controlli (descrittivi, senza soglia): rho(k) calcolato con una frase di E_ood al posto
di X_forget; rho_rand(k); risultati con rcond 1e-3 e 1e-9.

## INTERPRETAZIONE (fissata ora)

- H1 e H2 vere: la protezione multi-vincolo è promettente; passo successivo, versione
  iterativa. Non è una prova di forgetting riuscito.
- H1 falsa: proteggere un insieme non protegge le correlate non note; il collaterale è
  un problema di copertura.
- H1 vera e H2 falsa: la protezione funziona ma consuma il segnale di forgetting
  (entanglement al primo ordine).
- H3 falsa: l'analisi al primo ordine non spiega il regime dei passi necessari; nessuna
  conclusione sulla retain-invariance al primo ordine vale qui.
- Molti stati censurati o non applicabili: esito non conclusivo, da dichiarare.
Ogni esito si riporta con lo stesso dettaglio, positivo o negativo.
