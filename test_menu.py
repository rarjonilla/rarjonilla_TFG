import os


def mostrar_menu():
    print("1. Supervised Training - Round points")
    print("2. Supervised Training - Win or Lose")
    print("3. Supervised Training - Round points + heuristic")
    print("4. All Supervised Training from same generated data")


def ingresar_valor_entero():
    while True:
        try:
            valor = int(input("Ingrese un valor entero: "))
            print("Ha ingresado el valor:", valor)
            break
        except ValueError:
            print("Por favor, ingrese un valor entero válido.")


def mostrar_contenido_carpeta():
    carpeta = input("Ingrese la ruta de la carpeta: ")
    if os.path.isdir(carpeta):
        archivos = os.listdir(carpeta)
        print("Archivos en la carpeta:")
        for i, archivo in enumerate(archivos, start=1):
            print(f"{i}. {archivo}")

        while True:
            try:
                opcion = int(input("Elija un número correspondiente al archivo: "))
                if 1 <= opcion <= len(archivos):
                    archivo_elegido = archivos[opcion - 1]
                    print("Ha elegido el archivo:", archivo_elegido)
                    break
                else:
                    print("Por favor, elija un número válido.")
            except ValueError:
                print("Por favor, ingrese un número válido.")
    else:
        print("La ruta ingresada no corresponde a una carpeta.")


while True:
    mostrar_menu()
    opcion = input("Ingrese su opción: ")

    if opcion == "1":
        ingresar_valor_entero()
    elif opcion == "2":
        mostrar_contenido_carpeta()
    elif opcion == "3":
        print("Saliendo del programa...")
        break
    else:
        print("Opción no válida. Por favor, ingrese una opción válida.")