
def load_params(file):
    lines = open(file).readlines()
    dic = {}
    for line in lines:
        itemList = line.split()
        dic[itemList[0]] = float(itemList[1])
    return dic
