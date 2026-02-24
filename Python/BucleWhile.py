import math

numero = int(input("Digite un numero: "))

while numero<0: 
    print ("Error -> Debería ser un numero positivo ")
    numero = int(input("Digite un numero: "))

print(f"\nSu raíz cuadrada es: {(math.sqrt(numero)):.2f}")