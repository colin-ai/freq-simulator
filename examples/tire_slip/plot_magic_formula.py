import matplotlib.pyplot as plt

from simulator.model import magic_formula

# Model parameters
m = 300  # mass
g = 9.81  # gravity constant
p = m * g  # weight

F, k = magic_formula(p)

fig, ax = plt.subplots(1)
ax.plot(k, F)
ax.set(
    xlabel="alpha",
    ylabel="$F_{longitudinal}$",
    title="Courbe de la Magic Formula \n " "Avec m=300, b=10, c=1.9, d=1, e=0.97",
)
plt.show()
