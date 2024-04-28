from itertools import product


def calc_intervals_points_combination(is_brisca: bool, num_players: int) -> list[int]:
    # Valors disponibles per a sumar
    values = [0, 2, 3, 4, 10, 11]

    # Generar combinacions possibles de sumands
    combinations = product(values, repeat=num_players)

    # Calcular les sumes úniques
    sums = set(sum(comb) for comb in combinations)

    # Calcular les sumes diferents en sumar 10 al resultat
    other_sums = set(one_sum + 10 for one_sum in sums)

    all_sums = sums.union(other_sums)

    if not is_brisca:
        other_sums_tute_1 = set(one_sum + 40 for one_sum in sums)
        other_sums_tute_2 = set(one_sum + 20 for one_sum in sums)
        other_sums_tute_3 = set(one_sum + 50 for one_sum in sums)

        all_sums = all_sums.union(other_sums_tute_1)
        all_sums = all_sums.union(other_sums_tute_2)
        all_sums = all_sums.union(other_sums_tute_3)

    all_negative_sums = set(-one_sum for one_sum in all_sums)
    all_sums = all_sums.union(all_negative_sums)

    # Imprimir las sumas únicas
    print("Brisca, {} players:".format(num_players))
    # for points in sorted(all_sums):
    print(sorted(all_sums))
    print(len(all_sums))

    return sorted(all_sums)

calc_intervals_points_combination(True, 2)