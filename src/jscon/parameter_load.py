
def load_params(file):
    lines = open(file).readlines()
    dic = {}
    for line in lines:
        itemList = line.split()
        try:
            dic[itemList[0]] = float(itemList[1])
        except:
            dic[itemList[0]] = str(itemList[1])
    return dic
