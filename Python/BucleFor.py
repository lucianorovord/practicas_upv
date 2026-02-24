array1 = {"Alejandro":23, "Maria":22, "Esteban":33, "Luis":45, "Juan Carlos":17}

for i in array1: 
    print(f"{array1} -> {i}")

# OTRA FORMA DE IMPLEMENTARLO 

array1 = {"Alejandro":23, "Maria":22, "Esteban":33, "Luis":45, "Juan Carlos":17}

for clave,valor in array1.items(): 
    print(f"{clave} -> {valor}")