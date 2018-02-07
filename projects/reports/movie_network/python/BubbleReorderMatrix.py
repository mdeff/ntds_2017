def BubbleReorderMatrix(W,genres):
    def Intervals(genres):
        intervals =[0,
                     list(genres).count(0),
                     list(genres).count(0)+list(genres).count(1),
                     list(genres).count(0)+list(genres).count(1)+list(genres).count(2)]
        return intervals
    def SwapLines(W,index1,index2):
        newW = W[:]
        holder1 =W[index1]
        holder2 =W[index2]
        for i in range(0,len(Weights)):
            if i == index1:
                newW[i]=holder2
                continue
            if i == index2:
                newW[i]=holder1
                continue
        return newW
    def SymetricSwap(W,index1,index2,genres):
        holder = genres[index1]
        genres[index1]= genres[index2]
        genres[index2]= holder
        return  SwapLines((SwapLines(W,index1,index2).T),index1,index2),genres
    def GetIndexRegion(index,genres):
        intervals = Intervals(genres)
        if index < intervals[1]:
            return 0
        elif index < intervals[2]:
            return 1
        elif index < intervals[3]:
            return 2
        elif index >= intervals[3]:
            return 3
        return -1
        
    Weights=W[:]    
    print(genres[:30])
    for index in range(0,len(genres)):
        indexRegion = GetIndexRegion(index,genres)
        if indexRegion==genres[index]:
            continue
        for searcher in range(index+1,len(Weights)):
            if genres[index]==genres[searcher]:
                continue
            if genres[searcher]==indexRegion:
                #print("genres regions ",genres[index],indexRegion)
                Weights,genres=SymetricSwap(Weights,index,searcher,genres)
                #print("now genres regions ",genres[index],indexRegion)
    print(genres[:20])
    return W            