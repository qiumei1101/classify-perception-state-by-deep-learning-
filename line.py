from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


figure, ax = plt.subplots()

ax.set_xlim(left=0, right=20)
ax.set_ylim(bottom=0, top=10)

line1 = [(1, 1), (5, 5)]
line2 = [(11, 9), (8, 8)]
(line1_xs, line1_ys) = zip(*line1)
(line2_xs, line2_ys) = zip(*line2)

ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))
ax.add_line(Line2D(line2_xs, line2_ys, linewidth=1, color='red'))

plt.plot()
plt.show()
