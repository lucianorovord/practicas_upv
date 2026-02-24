num1 = int(input("Ejercicio 2 de Condicionales\nDigite el primer número: "))
num2 = int(input("Digite el segundo número: "))
num3 = int(input("Digite el tercer número: "))


if num1 > num2 and num1 > num3:
    print(f"El numero {num1} es mayor que : {num2} y {num3}")

elif num2 > num3:
    print(f"El numero {num2} es mayor que : {num1} y {num3}")

else:
    print(f"El numero {num3} es mayor que : {num1} y {num2}")