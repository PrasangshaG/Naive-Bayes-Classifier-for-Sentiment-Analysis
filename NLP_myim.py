import string

with open("C:\Users\prasangsha.ganguly\Downloads\\trainme.txt","r+") as traindata:
   array = []
   for line in traindata:
        array.append(line)



#mymat is a global matrix. 1st row: unique words; 2nd row: total count of the word in vocab; 3rd row: count of the word in positive docs; 4th row: count in negative docs

vocabpos = [None]*30000
vocabneg = [None]*30000
vocaball = [None]*50000
rows = 7
cols = 50000
mymat = [ ([0] * cols) for row in xrange(rows) ]

no_of_sentence = len(array)
uniquepos = 0
uniqueneg = 0


j = 0
k = 0
v = 0
Npos = 0
Nneg = 0
for it in range(0, no_of_sentence-1):

     p = array[it]
     st1 = str(p)
     st = string.lower(st1)
     s = st.translate(string.maketrans("",""), string.punctuation)
     #print s
     tokens = s.split()
     #print tokens[0]
     sizet = len(tokens)
     category = tokens[sizet - 1]
     i=0
     if(category == '1'):
       Npos = Npos + 1
       for i in range(0, sizet-1):
		 vocabpos[j] = tokens[i]
		 vocaball[v] = vocabpos[j]
		 j = j+1
		 i = i+1
		 v = v+1

     else:
       Nneg = Nneg + 1
       for i in range(0, sizet-1):
		 vocabneg[k] = tokens[i]
		 vocaball[v] = vocabneg[k]
		 k = k+1
		 i = i+1
		 v = v+1

#lvp is count of words in positive class, lvn is number of words in negative class
lvn =0
lvp =0
for i in range(0, 30000):
  if(vocabpos[i] != None):
    lvp = lvp +1
  if(vocabneg[i] != None):
    lvn = lvn +1    



#calculate prior p(+) = N(+)/ Ntot, Npos is number of positive sentences or documents but not words
Npos = float(Npos)
Nneg = float(Nneg)
ppos = Npos/no_of_sentence
pneg = Nneg/no_of_sentence


######## Create the first four rows of mymat ####################################
print("Please wait while we train the model")
l = 0
for i in range(0, v):
	temp = vocaball[i]
	for k in range(0, v):
		if(temp == mymat[0][k]):
			mymat[1][k] = mymat[1][k] + 1
			break
		else:
		    k = k+1
	if(k == v):
	   mymat[0][l] = temp
	   mymat[1][l] = 1
	   l = l+1
	   
for i in range(0, v):	   
	temppos = vocabpos[i]
	for k1 in range(0, v):
		if(temppos == mymat[0][k1]):
			if(mymat[2][k1] == 0):
				mymat[2][k1] = 1
			else:
			
			    mymat[2][k1] = mymat[2][k1] + 1
			break
		else:
		    k1 = k1 + 1
	tempneg = vocabneg[i]
	for k in range(0, v):
		if(tempneg == mymat[0][k]):
			if(mymat[3][k] == 0):
				mymat[3][k] = 1
			else:
				mymat[3][k] = mymat[3][k] + 1
			break
		else:
		    k = k+1	    	



#Now construct the posterior probability entries; 4th row: p(word|vocabulary); 5th row: p(word|positive); 6th row: p(word|negative) later smoothing is applied for
#5th and 6th row

for i in range(0, v):
   mymat[4][i] = float(mymat[1][i])/v
   mymat[5][i] = float(mymat[2][i])/lvp
   mymat[5][i] = float(mymat[2][i])/lvp
   mymat[6][i] = float(mymat[3][i])/lvn
	

print("Enter your smoothing technique choice")	
print("Enter 1 for Addone; 2 for Add-k; 3 for Jelinek Mercer; 4 for Dirichlet Prior; 5 for Absolute Discounting; 6 for Two Stage")
ch = input()

def addone():
	for i in range(0, v):
		mymat[5][i] = float(mymat[2][i] +1)/ (lvp + v)
		mymat[6][i] = float(mymat[3][i] +1)/ (lvn + v)

#addone()

def addk():
	k = input("Enter the additive factor K (0<K<1)")
	for i in range(0, v):
		mymat[5][i] = float(mymat[2][i] +k)/ (lvp + k*v)
		mymat[6][i] = float(mymat[3][i] +k)/ (lvn + k*v)



def JelinekMercer():
	lamb = input("Enter the smoothing parameter Lambda (0<=Lambda<=1)")
	for i in range(0, v):
		mymat[5][i] = float(mymat[2][i])*(1-lamb)/lvp + float(mymat[1][i])*lamb/v
		mymat[6][i] = float(mymat[3][i])*(1-lamb)/lvn + float(mymat[1][i])*lamb/v


#JelinekMercer()

def DirichletPrior():
	meu = input("Enter the dynamic coeffn M (0<=M<inf)")
	for i in range(0, v):
	    mymat[5][i] = (float(mymat[2][i]) + float(meu*mymat[4][i]))/(lvp + meu)
    	mymat[6][i] = (float(mymat[3][i]) + float(meu*mymat[4][i]))/(lvn + meu)

#DirichletPrior()


def AbsoluteDiscount():
	for i in range(0, v):
		if(mymat[2][i] != 0):
			global uniquepos
			uniquepos = uniquepos+1
		if(mymat[3][i] != 0):
			global uniqueneg
			uniqueneg = uniqueneg+1
	delta = input("Enter the discounting factor delta (0<=delta<=1)")
	for i in range(0, v):
		mymat[5][i] = (max(float(mymat[2][i])-delta, 0) + (delta*uniquepos*(float(mymat[1][i]))/v))/ lvp
		mymat[6][i] = (max(float(mymat[3][i])-delta, 0) + (delta*uniqueneg*(float(mymat[1][i]))/v))/ lvn
        
#AbsoluteDiscount()


def TwoStage():
	DirichletPrior()
	lamb = input("Enter Lambda")
	for i in range(0, v):
		tempp = float(mymat[5][i])
		tempn = float(mymat[6][i])
		mymat[5][i] = (1-lamb)*tempp + lamb*float(mymat[1][i])/v 
		mymat[6][i] = (1-lamb)*tempn + lamb*float(mymat[1][i])/v
		 
#TwoStage()

		  
if(ch == 1):
	addone()
if(ch == 2):
    addk()
elif(ch == 3):
    JelinekMercer()
elif(ch == 4):
    DirichletPrior()
elif(ch == 5):
	AbsoluteDiscount()
elif(ch == 6):
	TwoStage()
    	


#for i in range(0, v):
#	if(mymat[0][i] != 0):
#		print mymat[0][i]
#		print mymat[1][i]
#		print mymat[2][i]
#		print mymat[3][i]
#		print mymat[4][i]
#		print mymat[5][i]
#       print mymat[6][i]

finalpos1 = 1
finalneg1 = 1

Accuracy = 0.0


def Test():
	test = raw_input("Enter a test string ")
	s = test.translate(string.maketrans("",""), string.punctuation)
	test1 = string.lower(s)
	goku = test1.split()
	testlen = len(goku)
	pp = 1.0
	pn = 1.0
	for i in range(0, testlen):
		word = goku[i]
		for j in range(0, v):
			if(word == mymat[0][j]):
				pp = pp * float(mymat[5][j])
				pn = pn * float(mymat[6][j])
	finalpos =  ppos *pp
	finalneg =  pneg *pn
	if(finalpos >= finalneg):
		
		print("It is a positive statement")
	else:
		
		print("It is a negative statement")


#Test()


rows = 3
cols = 200
result = [ ([-1] * cols) for row in xrange(rows) ]


crow = 2
ccol = 2
confus =  [ ([0] * ccol) for row in xrange(crow)]


def Testdoc():
	with open("C:\Users\prasangsha.ganguly\Downloads\\testme1.txt","r+") as testdata:
		arraytest = []
		for line in testdata:
			arraytest.append(line)
	testlen = len(arraytest)
	for it in range(0, testlen-1):
		p = arraytest[it]
		st1 = str(p)
		st = string.lower(st1)
		s = st.translate(string.maketrans("",""), string.punctuation)
		result[0][it] = s
		token = s.split()
		sizet = len(token)
		category = token[sizet - 1]
		result[1][it] = category
		ppt = 1.0
		pnt = 1.0
		for i in range(0, sizet-1):
			word = token[i]
			for j in range(0, v):
				if(word == mymat[0][j]):
					ppt = ppt * float(mymat[5][j])
					pnt = pnt * float(mymat[6][j])
		finalpos1 = ppos * ppt
		finalneg1 = pneg * pnt
		if(finalpos1 >= finalneg1):
			result[2][it] = 1
		else:
			result[2][it] = 0
	for i in range(0, testlen):
		if(int(result[1][i]) == 1 and int(result[2][i]) == 1):
			confus[0][0] = confus[0][0] +1
		elif(int(result[1][i]) == 0 and int(result[2][i]) == 0):
			confus[1][1] = confus[1][1] + 1
		elif(int(result[1][i]) == 1 and int(result[2][i]) == 0):
			confus[0][1] = confus[0][1] +1
		else:
			confus[1][0] = confus[1][0] +1
	Accuracy = 	float(confus[0][0] + confus[1][1]) / testlen	
	print("Accuracy is:", Accuracy)

		
#Testdoc()

print("\n")

print("Enter 0 if you want to test sentiment for a single sentence")
print("Enter 1 if you want to test some document already labeled and find accuracy")
choice = input()


if(choice == 0):
	Test()
elif(choice == 1):
	Testdoc()



    	 