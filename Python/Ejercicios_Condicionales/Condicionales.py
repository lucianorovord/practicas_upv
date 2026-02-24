numero = int(input("Digite un numero: "))
Numero = int(input("Digite otro numero: "))

nums_pares = 0

if numero%2==0 and Numero%2==0:
    print("Ambos números son PARES")
elif numero%2==0:
    print(f"El numero: {numero} es PAR")
elif Numero %2==0:
    print(f"El numero: {Numero} es PAR")
else:
    print("Ninguno es par")
