import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pandas import DataFrame


class Generate_results_graphs:

    def __init__(self, csv_file: str, num_players: int, single_mode: bool, type: str):
        # Informació per generar els CSV
        self.num_players: int = num_players
        self.single_mode: int = single_mode
        self.type = type
        self.show = False

        # Es carrega el dataframe del CSV
        self.df: DataFrame = pd.read_csv(csv_file, dtype={'Rules': str})

        # Es genera el gràfic de barra i el de doble anella
        self.pairings_wins_double_bar()
        if self.num_players == 2:
            self.pairings_wins_double_bar_sl()

        self.total_points_wins_ring()
        if self.num_players == 2:
            self.total_points_wins_ring_sl()

    def pairings_wins_double_bar(self):
        # Es mostra un gràfic de barres per victòries de cada emparellament
        if self.num_players == 2:
            df_filtered = self.df[self.df['Rules'] == "0000"]
        else:
            df_filtered = self.df

        # Mida del gràfic
        fig, ax = plt.subplots(figsize=(10, 8))

        # Ample de les barres
        if self.num_players == 2 or not self.single_mode:
            bar_width = 0.35
        elif self.num_players == 3:
            bar_width = 0.25
        elif self.num_players == 4:
            bar_width = 0.15

        # Llista de tots els models del dataset
        if self.num_players == 2:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']))
        elif self.num_players == 3:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']) | set(df_filtered['Player_2']))
        elif self.num_players == 4:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']) | set(df_filtered['Player_2']) | set(df_filtered['Player_3']))

        # Posicions de les barres
        bar_pos = [np.arange(len(df_filtered))]
        if self.single_mode:
            for i in range(1, self.num_players):
                bar_pos.append([x + bar_width for x in bar_pos[i - 1]])
        else:
            bar_pos.append([x + bar_width for x in bar_pos[0]])

        # Colors per els models
        models_color = ['blue', 'green', 'red', 'orange']
        colors = {}
        for idx, model in enumerate(models):
            colors[model] = models_color[idx]

        bars = []
        if self.single_mode:
            for i in range(0, self.num_players):
                # Crear barres per a cada jugador
                bars.append(ax.bar(bar_pos[i], df_filtered[f'Wins_p{i}'], color=[colors[model] for model in df_filtered[f'Player_{i}']], width=bar_width, edgecolor='grey', label=f'Partides guanyades jugador {i}'))
        else:
            for i in range(0, self.num_players // 2):
                # Crear barres per a cada equip
                bars.append(ax.bar(bar_pos[i], df_filtered[f'Wins_p{i}'], color=[colors[model] for model in df_filtered[f'Player_{i}']], width=bar_width,edgecolor='grey', label=f'Partides guanyades jugador {i}'))

        # Etiquetes i títols
        ax.set_xlabel('Emparellaments', fontweight='bold')
        ax.set_ylabel('Partides guanyades', fontweight='bold')
        ax.set_title('Partides guanyades per emparellament')
        # ax.set_xticks([r + bar_width / 2 for r in range(len(self.df))])

        #         if self.num_players == 2:
        #             ax.set_xticklabels([f"{row['Player_0']} vs {row['Player_1']}" for _, row in self.df.iterrows()])
        #         elif self.num_players == 3:
        #             ax.set_xticklabels([f"{row['Player_0']} vs {row['Player_1']} vs {row['Player_2']}" for _, row in self.df.iterrows()])
        #         elif self.num_players == 4:
        #             if self.single_mode:
        #                 ax.set_xticklabels([f"{row['Player_0']} vs {row['Player_1']} vs {row['Player_2']} vs {row['Player_3']}" for _, row in self.df.iterrows()])
        #             else:
        #                 ax.set_xticklabels([f"{row['Player_0']} / {row['Player_2']}  vs {row['Player_1']} / {row['Player_3']}" for _, row in self.df.iterrows()])

        # Llegenda
        legend_patches = [Patch(color=color, label=model) for model, color in colors.items()]
        ax.legend(handles=legend_patches, loc='upper left')

        # Emmagatzemar i mostrar el gràfic
        plt.savefig(f'{self.type}_pairings_{self.num_players}_players_{self.single_mode}.png')
        if self.show:
            plt.show()

    def pairings_wins_double_bar_sl(self):
        # Es mostra un gràfic de barres per victòries de cada emparellament
        df_filtered = self.df[~self.df['Player_0'].isin(['rl', 'ga']) & ~self.df['Player_1'].isin(['rl', 'ga'])]

        # Mida del gràfic
        fig, ax = plt.subplots(figsize=(10, 8))

        # Ample de les barres
        if self.num_players == 2 or not self.single_mode:
            bar_width = 0.35
        elif self.num_players == 3:
            bar_width = 0.25
        elif self.num_players == 4:
            bar_width = 0.15

        # Llista de tots els models del dataset
        if self.num_players == 2:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']))
        elif self.num_players == 3:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']) | set(df_filtered['Player_2']))
        elif self.num_players == 4:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']) | set(df_filtered['Player_2']) | set(df_filtered['Player_3']))

        # Posicions de les barres
        bar_pos = [np.arange(len(df_filtered))]
        if self.single_mode:
            for i in range(1, self.num_players):
                bar_pos.append([x + bar_width for x in bar_pos[i - 1]])
        else:
            bar_pos.append([x + bar_width for x in bar_pos[0]])

        # Colors per els models
        models_color = ['blue', 'green', 'red', 'orange']
        colors = {}
        for idx, model in enumerate(models):
            colors[model] = models_color[idx]

        bars = []
        for i in range(0, self.num_players):
            # Crear barres per a cada jugador
            bars.append(ax.bar(bar_pos[i], df_filtered[f'Wins_p{i}'], color=[colors[model] for model in df_filtered[f'Player_{i}']], width=bar_width, edgecolor='grey', label=f'Partides guanyades jugador {i}'))

        # Etiquetes i títols
        ax.set_xlabel('Emparellaments segons regles', fontweight='bold')
        ax.set_ylabel('Partides guanyades', fontweight='bold')
        ax.set_title('Partides guanyades per emparellament')
        ax.set_xticks([r + bar_width / 2 for r in range(len(df_filtered))])
        ax.set_xticklabels([f"{row['Rules']}" for _, row in df_filtered.iterrows()])

        #         if self.num_players == 2:
        #             ax.set_xticklabels([f"{row['Player_0']} vs {row['Player_1']}" for _, row in self.df.iterrows()])
        #         elif self.num_players == 3:
        #             ax.set_xticklabels([f"{row['Player_0']} vs {row['Player_1']} vs {row['Player_2']}" for _, row in self.df.iterrows()])
        #         elif self.num_players == 4:
        #             if self.single_mode:
        #                 ax.set_xticklabels([f"{row['Player_0']} vs {row['Player_1']} vs {row['Player_2']} vs {row['Player_3']}" for _, row in self.df.iterrows()])
        #             else:
        #                 ax.set_xticklabels([f"{row['Player_0']} / {row['Player_2']}  vs {row['Player_1']} / {row['Player_3']}" for _, row in self.df.iterrows()])

        # Llegenda
        legend_patches = [Patch(color=color, label=model) for model, color in colors.items()]
        ax.legend(handles=legend_patches, loc='upper left')

        # Emmagatzemar i mostrar el gràfic
        plt.savefig(f'{self.type}_pairings_{self.num_players}_players_{self.single_mode}_rules.png')
        if self.show:
            plt.show()

    def total_points_wins_ring(self):
        # Es mostra un gràfic de doble anella, l'exterior amb el total de victòries de cada model per a tots els emparellaments i el segon amb el total de punts

        # Només per print dataframe
        # Mostrar totes les columnes
        pd.set_option('display.max_columns', None)
        # Mostrar tot el contingut de cada cel·la
        pd.set_option('display.max_colwidth', None)
        # Ajustar l'ample de la fila
        pd.set_option('display.width', None)

        if self.single_mode and self.num_players == 2:
            df_filtered = self.df[~self.df['Player_0'].isin(['rl', 'ga']) & ~self.df['Player_1'].isin(['rl', 'ga'])]
        else:
            df_filtered = self.df

        # Calcular el total de partides guanyades i punts de cada model
        totals = pd.DataFrame(columns=['Model', 'Wins', 'Points'])

        # Llista de tots els models del dataset
        if self.num_players == 2:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']))
        elif self.num_players == 3:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']) | set(df_filtered['Player_2']))
        elif self.num_players == 4:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']) | set(df_filtered['Player_2']) | set(df_filtered['Player_3']))

        # Calcular els totals
        for model in models:
            if self.num_players == 2 or not self.single_mode:
                total_wins = df_filtered[df_filtered['Player_0'] == model]['Wins_p0'].sum() + df_filtered[df_filtered['Player_1'] == model]['Wins_p1'].sum()
                total_points = df_filtered[df_filtered['Player_0'] == model]['Points_p0'].sum() + df_filtered[df_filtered['Player_1'] == model]['Points_p1'].sum()
            elif self.num_players == 3:
                total_wins = df_filtered[df_filtered['Player_0'] == model]['Wins_p0'].sum() + df_filtered[df_filtered['Player_1'] == model]['Wins_p1'].sum() + df_filtered[df_filtered['Player_2'] == model]['Wins_p2'].sum()
                total_points = df_filtered[df_filtered['Player_0'] == model]['Points_p0'].sum() + df_filtered[df_filtered['Player_1'] == model]['Points_p1'].sum() + df_filtered[df_filtered['Player_2'] == model]['Points_p2'].sum()
            elif self.num_players == 4:
                total_wins = df_filtered[df_filtered['Player_0'] == model]['Wins_p0'].sum() + df_filtered[df_filtered['Player_1'] == model]['Wins_p1'].sum() + df_filtered[df_filtered['Player_2'] == model]['Wins_p2'].sum() + df_filtered[df_filtered['Player_3'] == model]['Wins_p3'].sum()
                total_points = df_filtered[df_filtered['Player_0'] == model]['Points_p0'].sum() + df_filtered[df_filtered['Player_1'] == model]['Points_p1'].sum() + df_filtered[df_filtered['Player_2'] == model]['Points_p2'].sum() + df_filtered[df_filtered['Player_3'] == model]['Points_p3'].sum()
            # totals = totals.append({'Model': model, 'Wins': total_wins, 'Points': total_points}, ignore_index=True)
            totals = pd.concat([totals, pd.DataFrame([{'Model': model, 'Wins': total_wins, 'Points': total_points}])], ignore_index=True)

        print(df_filtered)
        print(totals)

        # Colors dels models
        colors_wins = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        colors_points = ['blue', 'green', 'red', 'orange']

        # Crear el gràfico d'anell doble
        fig, ax = plt.subplots(figsize=(8, 8))

        # Primer anell (partides guanyades)
        wedges1, texts1 = ax.pie(totals['Wins'], labels=totals['Model'], startangle=90, colors=colors_wins, radius=1.2, wedgeprops=dict(width=0.3, edgecolor='w'))

        # Segon anell (punts)
        wedges2, texts2 = ax.pie(totals['Points'], startangle=90, colors=colors_points, radius=0.9, wedgeprops=dict(width=0.3, edgecolor='w'))

        # Afegir els totals centrats a la porció de l'anell
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
        # ax.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
        ax.legend(legend_handles, legend_labels, loc='upper left')

        # Emmagatzemar i mostrar el gràfic
        plt.savefig(f'{self.type}_total_wins_points_{self.num_players}_players_{self.single_mode}.png')
        if self.show:
            plt.show()

    def total_points_wins_ring_sl(self):
        # Es mostra un gràfic de doble anella, l'exterior amb el total de victòries de cada model per a tots els emparellaments i el segon amb el total de punts
        if self.num_players == 2:
            df_filtered = self.df[self.df['Rules'] == "0000"]
        else:
            df_filtered = self.df

        # Calcular el total de partides guanyades i punts de cada model
        totals = pd.DataFrame(columns=['Model', 'Wins', 'Points'])

        # Llista de tots els models del dataset
        if self.num_players == 2:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']))
        elif self.num_players == 3:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']) | set(df_filtered['Player_2']))
        elif self.num_players == 4:
            models = list(set(df_filtered['Player_0']) | set(df_filtered['Player_1']) | set(df_filtered['Player_2']) | set(
                df_filtered['Player_3']))

        # Calcular els totals
        for model in models:
            if self.num_players == 2 or not self.single_mode:
                total_wins = df_filtered[df_filtered['Player_0'] == model]['Wins_p0'].sum() + \
                             df_filtered[df_filtered['Player_1'] == model]['Wins_p1'].sum()
                total_points = df_filtered[df_filtered['Player_0'] == model]['Points_p0'].sum() + \
                               df_filtered[df_filtered['Player_1'] == model]['Points_p1'].sum()
            elif self.num_players == 3:
                total_wins = df_filtered[df_filtered['Player_0'] == model]['Wins_p0'].sum() + \
                             df_filtered[df_filtered['Player_1'] == model]['Wins_p1'].sum() + \
                             df_filtered[df_filtered['Player_2'] == model]['Wins_p2'].sum()
                total_points = df_filtered[df_filtered['Player_0'] == model]['Points_p0'].sum() + \
                               df_filtered[df_filtered['Player_1'] == model]['Points_p1'].sum() + \
                               df_filtered[df_filtered['Player_2'] == model]['Points_p2'].sum()
            elif self.num_players == 4:
                total_wins = df_filtered[df_filtered['Player_0'] == model]['Wins_p0'].sum() + \
                             df_filtered[df_filtered['Player_1'] == model]['Wins_p1'].sum() + \
                             df_filtered[df_filtered['Player_2'] == model]['Wins_p2'].sum() + \
                             df_filtered[df_filtered['Player_3'] == model]['Wins_p3'].sum()
                total_points = df_filtered[df_filtered['Player_0'] == model]['Points_p0'].sum() + \
                               df_filtered[df_filtered['Player_1'] == model]['Points_p1'].sum() + \
                               df_filtered[df_filtered['Player_2'] == model]['Points_p2'].sum() + \
                               df_filtered[df_filtered['Player_3'] == model]['Points_p3'].sum()
            # totals = totals.append({'Model': model, 'Wins': total_wins, 'Points': total_points}, ignore_index=True)
            totals = pd.concat([totals, pd.DataFrame([{'Model': model, 'Wins': total_wins, 'Points': total_points}])],ignore_index=True)

        # Colors dels models
        colors_wins = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        colors_points = ['blue', 'green', 'red', 'orange']

        # Crear el gràfico d'anell doble
        fig, ax = plt.subplots(figsize=(8, 8))

        # Primer anell (partides guanyades)
        wedges1, texts1 = ax.pie(totals['Wins'], labels=totals['Model'], startangle=90, colors=colors_wins, radius=1.2,
                                 wedgeprops=dict(width=0.3, edgecolor='w'))

        # Segon anell (punts)
        wedges2, texts2 = ax.pie(totals['Points'], startangle=90, colors=colors_points, radius=0.9,
                                 wedgeprops=dict(width=0.3, edgecolor='w'))

        # Afegir els totals centrats a la porció de l'anell
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
        # ax.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
        ax.legend(legend_handles, legend_labels, loc='upper left')

        # Emmagatzemar i mostrar el gràfic
        plt.savefig(f'{self.type}_total_wins_points_{self.num_players}_players_{self.single_mode}_rules.png')
        if self.show:
            plt.show()


doThis = False
if doThis:
    Generate_results_graphs('brisca_2j.csv', 2, True, 'brisca')
    Generate_results_graphs('brisca_3j.csv', 3, True, 'brisca')
    Generate_results_graphs('brisca_4j.csv', 4, True, 'brisca')
    Generate_results_graphs('brisca_4jt.csv', 4, False, 'brisca')
    Generate_results_graphs('tute_2j.csv', 2, True, 'tute')
    Generate_results_graphs('tute_3j.csv', 3, True, 'tute')
    Generate_results_graphs('tute_4j.csv', 4, True, 'tute')
    Generate_results_graphs('tute_4jt.csv', 4, False, 'tute')
    Generate_results_graphs('tute_assist_2j.csv', 2, True, 'tute_oa')
    Generate_results_graphs('tute_assist_3j.csv', 3, True, 'tute_oa')
    Generate_results_graphs('tute_assist_4j.csv', 4, True, 'tute_oa')
    Generate_results_graphs('tute_assist_4jt.csv', 4, False, 'tute_oa')