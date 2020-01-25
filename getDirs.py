# Python program to explain os.listdir() method  
    
# importing os module  
import os 
import random
  
# Get the path of current working directory 
path = os.getcwd() 
  
# Get the list of all files and directories 
# in current working directory 
dir_list = os.listdir('/local/2020_hackathon/2020_hackathon/') 
length = len(dir_list)  
  
#iprint("Files and directories in (len: "+str(length)+"'", path, "' :")  
# print the list 
# print(dir_list)


print("----------")
data = []
for i in range(length):
    data.append(i)
print(data)
random.shuffle(data)
print("##################################")
print(data)

eighty_list = []
twenty_list = []
eighty_list_num = data[:6927]
twenty_list_num = data[6928:]
for i in eighty_list_num:
    eighty_list.append(dir_list[i])
for i in twenty_list_num:
    twenty_list.append(dir_list[i])
print("80 list *********************************")
print(eighty_list)
print("20 list *********************************")
print(twenty_list)

f = open('eighty_list.txt', 'r+')
f.truncate(0) # need '0' when using r+
f.close()



f = open("eighty_list.txt", "a")
for i in range(len(eighty_list)):
    f.write(str(eighty_list[i])+'\n')
f.close()


f = open('twenty_list.txt', 'r+')
f.truncate(0) # need '0' when using r+
f.close()



f = open("twenty_list.txt", "a")
for i in range(len(twenty_list)):
    f.write(str(twenty_list[i])+'\n')
f.close()
