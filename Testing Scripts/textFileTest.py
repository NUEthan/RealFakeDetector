# If file DNE then create it, else appending data to it.
f = open("test.txt", 'a')
f.write("New Line\n")
f.close()
