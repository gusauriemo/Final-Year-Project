import numpy as np
import itertools

notes = range(88)
notes_plus_1 = [x+1 for x in notes]
eighty8c2 = []#88 choose 2 array
for L in range(2, 3): 
    for subset in itertools.combinations(notes_plus_1, L):
        eighty8c2.append(subset)
print(eighty8c2)

intervals = [] #array of interval values with the same indexes as the 88c2 array
for i in range(3828):
    intervals.append(abs(eighty8c2[i][0]- eighty8c2[i][1]))

noo=np.array([intervals]) # array created so that we can use np.where


index1=((np.where(noo==1)[1]))
index2=((np.where(noo==2)[1]))
index3=((np.where(noo==3)[1]))
index4=((np.where(noo==4)[1]))
index5=((np.where(noo==5)[1]))
index6=((np.where(noo==6)[1]))
index7=((np.where(noo==7)[1]))
index8=((np.where(noo==8)[1]))
index9=((np.where(noo==9)[1]))
index10=((np.where(noo==10)[1]))
index11=((np.where(noo==11)[1]))
index12=((np.where(noo==12)[1]))
index13=((np.where(noo==13)[1]))
index14=((np.where(noo==14)[1]))
index15=((np.where(noo==15)[1]))
index16=((np.where(noo==16)[1]))
index17=((np.where(noo==17)[1]))
index18=((np.where(noo==18)[1]))
index19=((np.where(noo==19)[1]))
index20=((np.where(noo==20)[1]))
index21=((np.where(noo==21)[1]))
index22=((np.where(noo==22)[1]))
index23=((np.where(noo==23)[1]))
index24=((np.where(noo==24)[1]))
index25=((np.where(noo==25)[1]))
index26=((np.where(noo==26)[1]))
index27=((np.where(noo==27)[1]))
index28=((np.where(noo==28)[1]))
index29=((np.where(noo==29)[1]))
index30=((np.where(noo==30)[1]))
index31=((np.where(noo==31)[1]))
index32=((np.where(noo==32)[1]))
index33=((np.where(noo==33)[1]))
index34=((np.where(noo==34)[1]))
index35=((np.where(noo==35)[1]))
index36=((np.where(noo==36)[1]))
index37=((np.where(noo==37)[1]))
index38=((np.where(noo==38)[1]))
index39=((np.where(noo==39)[1]))
index40=((np.where(noo==40)[1]))
index41=((np.where(noo==41)[1]))
index42=((np.where(noo==42)[1]))
index43=((np.where(noo==43)[1]))
index44=((np.where(noo==44)[1]))
index45=((np.where(noo==45)[1]))
index46=((np.where(noo==46)[1]))
index47=((np.where(noo==47)[1]))
index48=((np.where(noo==48)[1]))
index49=((np.where(noo==49)[1]))
index50=((np.where(noo==50)[1]))
index51=((np.where(noo==51)[1]))
index52=((np.where(noo==52)[1]))
index53=((np.where(noo==53)[1]))
index54=((np.where(noo==54)[1]))
index55=((np.where(noo==55)[1]))
index56=((np.where(noo==56)[1]))
index57=((np.where(noo==57)[1]))
index58=((np.where(noo==58)[1]))
index59=((np.where(noo==59)[1]))
index60=((np.where(noo==60)[1]))
index61=((np.where(noo==61)[1]))
index62=((np.where(noo==62)[1]))
index63=((np.where(noo==63)[1]))
index64=((np.where(noo==64)[1]))
index65=((np.where(noo==65)[1]))
index66=((np.where(noo==66)[1]))
index67=((np.where(noo==67)[1]))
index68=((np.where(noo==68)[1]))
index69=((np.where(noo==69)[1]))
index70=((np.where(noo==70)[1]))
index71=((np.where(noo==71)[1]))
index72=((np.where(noo==72)[1]))
index73=((np.where(noo==73)[1]))
index74=((np.where(noo==74)[1]))
index75=((np.where(noo==75)[1]))
index76=((np.where(noo==76)[1]))
index77=((np.where(noo==77)[1]))
index78=((np.where(noo==78)[1]))
index79=((np.where(noo==79)[1]))
index80=((np.where(noo==80)[1]))
index81=((np.where(noo==81)[1]))
index82=((np.where(noo==82)[1]))
index83=((np.where(noo==83)[1]))
index84=((np.where(noo==84)[1]))
index85=((np.where(noo==85)[1]))
index86=((np.where(noo==86)[1]))
index87=((np.where(noo==87)[1]))


names=[] #array with all the indexes
for i in range(88): 
    names.append("index"+str(i))
names.pop(0) #take off the element called index0

wheres = [] # array with all the names indexed
for i in range(87):
    wheres.append(globals()[names[i]])

def third_layer_connections(interval):
    """The following function outputs all the pairings for any given interval.
    Effectively, it maps the neurons from second layer into third layer.
    E.g., for interval=87, the output is [(1, 88)]
    """
    results = []
    for i in range(interval-1,interval):
        for j in range(87-i):
            results.append(eighty8c2[wheres[i][j]])
    return results

print(third_layer_connections(87))



