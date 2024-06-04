import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pandas import DataFrame


class Generate_results_graphs:

    def __init__(self, csv_file: str, num_players: int, single_mode: bool):
        self.num_players: int = num_players
        self.single_mode: int = single_mode
        # csv_file = "game_resultats.csv"
        self.df: DataFrame = pd.read_csv(csv_file)

        self.pairings_wins_points_double_bar()
        self.total_points_wins_ring()

    def pairings_wins_points_double_bar(self):
        # Mida del gràfic
        fig, ax = plt.subplots(figsize=(14, 7))

        # Ample de les barres
        if self.num_players == 2 or not self.single_mode:
            bar_width = 0.35
        elif self.num_players == 3:
            bar_width = 0.25
        elif self.num_players == 4:
            bar_width = 0.15

        # Posicions de les barres
        bar_pos = [np.arange(len(self.df))]
        if self.single_mode:
            for i in range(1, self.num_players):
                bar_pos.append([x + bar_width for x in bar_pos[i - 1]])
        else:
            bar_pos.append([x + bar_width for x in bar_pos[0]])

            # Llista de tots els models del dataset
            if self.num_players == 2:
                models = list(set(self.df['Player_0']) | set(self.df['Player_1']))
            elif self.num_players == 3:
                models = list(set(self.df['Player_0']) | set(self.df['Player_1']) | set(self.df['Player_2']))
            elif self.num_players == 4:
                models = list(set(self.df['Player_0']) | set(self.df['Player_1']) | set(self.df['Player_2']) | set(self.df['Player_3']))

        # Colors per els models
        models_color = ['blue', 'green', 'red', 'orange']
        colors = {}
        for idx, model in enumerate(models):
            colors[model] = models_color[idx]

        bars = []
        for i in range(0, self.num_players):
            # Crear barres per a cada jugador
            bars.append(ax.bar(bar_pos[i], self.df[f'Wins_p{i}'], color=[colors[model] for model in self.df[f'Player_{i}']], width=bar_width, edgecolor='grey', label=f'Partides guanyades jugador {i}'))

        # Etiquetes i títols
        ax.set_xlabel('Emparellaments', fontweight='bold')
        ax.set_ylabel('Partides guanyades', fontweight='bold')
        ax.set_title('Partides guanyades i punts per emparellament')
        ax.set_xticks([r + bar_width / 2 for r in range(len(self.df))])

        if self.num_players == 2:
            ax.set_xticklabels([f"{row['Player_0']} vs {row['Player_1']}" for _, row in self.df.iterrows()])
        elif self.num_players == 3:
            ax.set_xticklabels([f"{row['Player_0']} vs {row['Player_1']} vs {row['Player_2']}" for _, row in self.df.iterrows()])
        elif self.num_players == 4:
            if self.single_mode:
                ax.set_xticklabels([f"{row['Player_0']} vs {row['Player_1']} vs {row['Player_2']} vs {row['Player_3']}" for _, row in self.df.iterrows()])
            else:
                ax.set_xticklabels([f"{row['Player_0']} / {row['Player_2']}  vs {row['Player_1']} / {row['Player_3']}" for _, row in self.df.iterrows()])

        # Llegenda personalitzada
        legend_patches = [Patch(color=color, label=modelo) for modelo, color in colors.items()]
        ax.legend(handles=legend_patches, loc='upper left')

        # Afegir puntuació
        for bars_ in bars:
            for bar in bars_:
                height = bar.get_height()
                ax.annotate('{}'.format(height),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        plt.savefig(f'pairings_{self.num_players}_players_{self.single_mode}.png')

        # Mostrar el gráfico
        plt.show()

    def total_points_wins_ring(self):
        # Calcular el total de partides guanyades i punts de cada model
        totals = pd.DataFrame(columns=['Model', 'Wins', 'Points'])

        # Llista de tots els models del dataset
        if self.num_players == 2:
            models = list(set(self.df['Player_0']) | set(self.df['Player_1']))
        elif self.num_players == 3:
            models = list(set(self.df['Player_0']) | set(self.df['Player_1']) | set(self.df['Player_2']))
        elif self.num_players == 4:
            models = list(set(self.df['Player_0']) | set(self.df['Player_1']) | set(self.df['Player_2']) | set(self.df['Player_3']))

        # Calcular els totals
        for model in models:
            if self.num_players == 2 or not self.single_mode:
                total_wins = self.df[self.df['Player_0'] == model]['Wins_p0'].sum() + self.df[self.df['Player_1'] == model]['Wins_p1'].sum()
                total_points = self.df[self.df['Player_0'] == model]['Points_p0'].sum() + self.df[self.df['Player_1'] == model]['Points_p1'].sum()
            elif self.num_players == 3:
                total_wins = self.df[self.df['Player_0'] == model]['Wins_p0'].sum() + self.df[self.df['Player_1'] == model]['Wins_p1'].sum() + self.df[self.df['Player_2'] == model]['Wins_p2'].sum()
                total_points = self.df[self.df['Player_0'] == model]['Points_p0'].sum() + self.df[self.df['Player_1'] == model]['Points_p1'].sum() + self.df[self.df['Player_2'] == model]['Points_p2'].sum()
            elif self.num_players == 4:
                total_wins = self.df[self.df['Player_0'] == model]['Wins_p0'].sum() + self.df[self.df['Player_1'] == model]['Wins_p1'].sum() + self.df[self.df['Player_2'] == model]['Wins_p2'].sum() + self.df[self.df['Player_3'] == model]['Wins_p3'].sum()
                total_points = self.df[self.df['Player_0'] == model]['Points_p0'].sum() + self.df[self.df['Player_1'] == model]['Points_p1'].sum() + self.df[self.df['Player_2'] == model]['Points_p2'].sum() + self.df[self.df['Player_3'] == model]['Points_p3'].sum()
            totals = totals.append({'Model': model, 'Wins': total_wins, 'Points': total_points}, ignore_index=True)

        # print(totals)

        # Colors dels models
        colors_wins = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        colors_points = ['blue', 'green', 'red', 'orange']

        # Crear el gràfico d'anell doble
        fig, ax = plt.subplots(figsize=(10, 7))

        # Primer anell (partides guanyades)
        wedges1, texts1 = ax.pie(totals['Wins'], labels=totals['Model'], startangle=90, colors=colors_wins, radius=1.2, wedgeprops=dict(width=0.3, edgecolor='w'))

        # Segundo anillo (puntos)
        wedges2, texts2 = ax.pie(totals['Points'], startangle=90, colors=colors_points, radius=0.9, wedgeprops=dict(width=0.3, edgecolor='w'))

        # Afegir els totals centrat en la porció de l'anell
        def get_wedge_center(wedge):
            theta1, theta2 = wedge.theta1, wedge.theta2
            center = wedge.center
            r = (wedge.r + wedge.r - wedge.width) / 2
            theta = (theta1 + theta2) / 2
            x = center[0] + r * np.cos(np.radians(theta))
            y = center[1] + r * np.sin(np.radians(theta))
            return x, y

        for i, (w1, w2) in enumerate(zip(wedges1, wedges2)):
            x1, y1 = get_wedge_center(w1)
            x2, y2 = get_wedge_center(w2)
            ax.text(x1, y1, f"{totals.iloc[i]['Wins']}", ha='center', va='center', color='black', fontweight='bold')
            ax.text(x2, y2, f"{totals.iloc[i]['Points']}", ha='center', va='center', color='white', fontweight='bold')

        # Títol
        ax.set_title('Puntuacions i partides guanyades per model')

        # Llegenda
        legend_labels = [f"{model} - Partides guanyades" for model in totals['Model']] + [f"{model} - Punts" for model in totals['Model']]
        legend_handles = wedges1 + wedges2
        ax.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1))

        plt.savefig(f'total_wins_points_{self.num_players}_players_{self.single_mode}.png')

        plt.show()
