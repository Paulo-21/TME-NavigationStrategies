import os
import matplotlib
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl

logname = ""
Qtable = {}
statetocoord = {}
with open("log/1762728364.7378485Qtable.txt", 'rb') as f:
    Qtable = pickle.load(f)
with open("log/1762728364.7378485StateToPos.txt", 'rb') as f:
    statetocoord = pickle.load(f)

vegetables = [i for i in range(0, 600)]
farmers = [i for i in range(0, 600)]

harvest = np.zeros((600,600))
for el in Qtable:
    if el == '':
        continue
    for (x, y) in statetocoord[el]:
        harvest[y, x] = int(  10 if Qtable[el][0] > Qtable[el][1] else -10)

fig, ax = plt.subplots()
im = ax.imshow(harvest)

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(len(farmers)), labels=farmers,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(vegetables)), labels=vegetables)

ax.set_title("")
fig.tight_layout()
plt.show()


all_tab_qlearning = []
for filename in os.listdir("log/"):
    if "Trial" in filename:
        val = []
        for line in open("log/"+filename):
            val.append(float(line))
        all_tab_qlearning.append(val)
plt.figure(figsize=(8, 5))
x = np.arange(len(all_tab_qlearning[0]))
##########################################################################
for k, j in enumerate(all_tab_qlearning):
    plt.plot(x, j,  marker='o', label='run n°'+str(k))

plt.title("Courbe de temps d'execution pour chaque run")
plt.xlabel("N°de l'essai")
plt.ylabel("seconde par essai")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
##################################################################################
# 10 premiers
all_tab_qlearning = np.array(all_tab_qlearning)
medianq = np.median(all_tab_qlearning[:10], axis=0)
q1q = np.percentile(all_tab_qlearning[:10], 25, axis=0)
q3q = np.percentile(all_tab_qlearning[:10], 75, axis=0)
# Derniers
medianq_last = np.median(all_tab_qlearning[-10:], axis=0)
q1q_last = np.percentile(all_tab_qlearning[-10:], 25, axis=0)
q3q_last = np.percentile(all_tab_qlearning[-10:], 75, axis=0)
x = np.arange(len(medianq_last))
plt.figure(figsize=(8, 5))
plt.fill_between(x, q1q, q3q, color='lightblue', alpha=0.4, label='Interquartile (Q1–Q3) 10 premiers')
plt.plot(x, medianq, color='orange', linestyle='--', marker='x', label='Médiane 10 premiers')
plt.fill_between(x, q1q_last, q3q_last, color='lightgreen', alpha=0.4, label='Interquartile (Q1–Q3) 10 derniers')
plt.plot(x, medianq_last, color='yellow', linestyle='--', marker='x', label='Médiane 10 derniers')
# --- Mise en forme ---
plt.title("Médiane et quartiles des 10 run sur les 10 premiers trial et les 10 derniers")
plt.xlabel("N°Trial")
plt.ylabel("Temps en seconde par trial")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#####################################
all_tab_randomPersist = []
for filename in os.listdir("log/"):
    if "random" in filename:
        val = []
        for line in open("log/"+filename):
            val.append(float(line))
        all_tab_randomPersist.append(val)

all_tab_randomPersist = np.array(all_tab_randomPersist)
mean = np.mean(all_tab_randomPersist, axis=0)
median = np.median(all_tab_randomPersist, axis=0)
q1 = np.percentile(all_tab_randomPersist, 25, axis=0)
q3 = np.percentile(all_tab_randomPersist, 75, axis=0)

x = np.arange(len(mean))
plt.figure(figsize=(8, 5))
plt.plot(x, median, color='blue', linestyle='--', marker='x', label='Médiane RandomPersist')

# Quartiles (en zone ombrée)
plt.fill_between(x, q1, q3, color='lightblue', alpha=0.4, label='Interquartile (Q1–Q3) RandomPersist')

# Médiane
plt.plot(x, medianq, color='red', linestyle='--', marker='x', label='Médiane Qlearning')

# Quartiles (en zone ombrée)
plt.fill_between(x, q1q, q3q, color='lightgreen', alpha=0.4, label='Interquartile (Q1–Q3) Qlearning')

# --- Mise en forme ---
plt.title("médiane et quartiles des 10 run entre Qleanring et RandomPersist")
plt.xlabel("N° Trial")
plt.ylabel("Seconde par épisode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
