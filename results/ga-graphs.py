import pandas as pd
import matplotlib.pyplot as plt


def more_wins():
    # Llegir el CSV del log del l'entrenament (partides guanyades i punts per a la simulació de cada individu a cada generació)
    path = '../ga_models/brisca/2j/0000_20240515_162533_10_games_1000_generations_layers_50_25_population_32_best_6_directive_1_elite_selection/game_log.csv'
    df = pd.read_csv(path)

    # Ordenar per Wins i Points el dataset
    df_sorted = df.sort_values(by=['Wins', 'Points'], ascending=False)

    # Es seleccionen les primeres 20 files
    df_top_20 = df_sorted.head(20)

    # Crear el gràfic de barres amb els models que més victòries han aconseguit i el total de punts
    plt.figure(figsize=(25, 6))

    # Barres per Wins
    plt.barh(df_top_20['Model name'], df_top_20['Wins'], color='blue', label='Victòries')

    # Afegir punts al final de la barra
    for i, (wins, points) in enumerate(zip(df_top_20['Wins'], df_top_20['Points'])):
        plt.text(wins + 0.02, i, str(round(points, 2)), va='center')

    # Afegir etiquetes i títol del gràfic
    plt.xlabel('Nom del model')
    plt.ylabel('Valors?')
    plt.title('Millors 20 individus per victòries i punts')
    plt.xticks(rotation=90)
    plt.legend()

    # Emmagatzemar i mostrar el gràfic
    plt.savefig('ga-20_millors_individus_per_victories_i_punts.png')
    plt.show()

def generations_progession():
    # Llegir el CSV del log del l'entrenament (partides guanyades i punts per a la simulació de cada individu a cada generació)
    path = '../ga_models/brisca/2j/0000_20240515_162533_10_games_1000_generations_layers_50_25_population_32_best_6_directive_1_elite_selection/game_log.csv'
    df = pd.read_csv(path)

    # Es vol els wins i points del millor individu cada 50 generacions (la 0, la 49, la 99, ..., la 999)
    # Extreure el número de generació de cada individu
    df['generation'] = df['Model name'].str.extract(r'generation_(\d+)_nn_\d+.keras').astype(int)

    # Seleccionar les generacions exactes cada 50
    generations_to_select = [i for i in range(1000) if i == 0 or (i + 1) % 50 == 0]
    df_filtered = df[df['generation'].isin(generations_to_select)]

    # Seleccionar el model amb més Wins per cada grup de generacions
    df_best_by_generation = df_filtered.groupby('generation').apply(lambda x: x.loc[x['Wins'].idxmax()])

    # Per poder ordenar els models per nom, tinc que afegir 0 a l'esquerra (0 -> 000, 10 -> 010)
    df_best_by_generation['Model name'] = df_best_by_generation.apply(lambda row: row['Model name'].split('_')[0] + '_' + row['Model name'].split('_')[1] + '_' + row['Model name'].split('_')[2].zfill(3) + '_' + row['Model name'].split('_')[3] + '_' + row['Model name'].split('_')[4], axis=1)

    # Ordenar per nom del model
    df_best_by_generation_sorted = df_best_by_generation.sort_values(by=['Model name'], ascending=False)

    # Crear el gràfic
    plt.figure(figsize=(22, 8))

    # Barres per Wins
    plt.barh(df_best_by_generation_sorted['Model name'], df_best_by_generation_sorted['Wins'], color='blue', label='Victòries')

    # Afegir punts al final de la barra
    for i, (wins, points) in enumerate(zip(df_best_by_generation_sorted['Wins'], df_best_by_generation_sorted['Points'])):
        plt.text(wins + 0.02, i, str(round(points, 2)), va='center')

    # Afegir etiquetes i títol del gràfic
    plt.xlabel('Victòries')
    plt.ylabel('Nom del model')
    plt.title('Millors individus per victòries i punts (cada 50 generacions)')
    plt.legend()

    # Emmagatzemar i mostrar el gràfic
    plt.savefig('ga-millor_individu_cada_50_generacions_per_victories_i_punts.png')
    plt.show()

def generations_median_wins():
    # Llegir el CSV del log del l'entrenament (partides guanyades i punts per a la simulació de cada individu a cada generació)
    df = pd.read_csv('../ga_models/brisca/2j/0000_20240515_162533_10_games_1000_generations_layers_50_25_population_32_best_6_directive_1_elite_selection/game_log.csv')

    # Es vol la mitja de victòries i punts cada 50 generacions (la 0, la 49, la 99, ..., la 999)

    # Extreure el número de generació de cada individu
    df['generation'] = df['Model name'].str.extract(r'generation_(\d+)_nn_\d+.keras').astype(int)

    # Seleccionar les generacions exactes cada 50
    generations_to_select = [i for i in range(1000) if i == 0 or (i + 1) % 50 == 0]
    df_filtered = df[df['generation'].isin(generations_to_select)]

    # Agrupar per el número de generació i calcular la mitjana de Wins i Points
    # df_grouped = df_filtered.groupby('generation').mean().reset_index()
    df_grouped = df_filtered.groupby('generation')[['Wins', 'Points']].mean().reset_index()

    # Crear un nou DataFrame amb les mitjanes
    df_new = pd.DataFrame(columns=df_grouped.columns)
    for generation in generations_to_select:
        # df_new = df_new.append(df_grouped[df_grouped['generation'] == generation])
        df_new = pd.concat([df_new, pd.DataFrame(df_grouped[df_grouped['generation'] == generation])], ignore_index=True)


    # Reindexar el nou DataFrame
    df_new.reset_index(drop=True, inplace=True)

    # Canviar el nom de les generacions perquè es puguin ordenar
    df_new['Model name'] = df_new['generation'].apply(lambda x: f'Generació {str(x).zfill(3)}')

    # Ordenar per nom del model
    df_best_by_generation_sorted = df_new.sort_values(by=['Model name'], ascending=False)

    # Crear el gràfic
    plt.figure(figsize=(22, 8))

    # Barres per Wins
    plt.barh(df_best_by_generation_sorted['Model name'], df_best_by_generation_sorted['Wins'], color='blue', label='Victòries')

    # Afegir punts al final de la barra
    for i, (wins, points) in enumerate(zip(df_best_by_generation_sorted['Wins'], df_best_by_generation_sorted['Points'])):
        plt.text(wins + 0.02, i, str(round(points, 2)), va='center')

    # Afegir etiquetes i títol del gràfic
    plt.xlabel('Victòries')
    plt.ylabel('Nom del model')
    plt.title('Mitjana de victòries i punts (cada 50 generacions)')
    plt.legend()

    # Emmagatzemar i mostrar el gràfic
    plt.savefig('ga-20_millor_individu_cada_50_generacions_per_victories_i_punts.png')
    plt.show()


more_wins()
# generations_progession()
# generations_median_wins()